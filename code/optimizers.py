import math
from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.optim import Optimizer

class ClusterCoupledAdam(Optimizer):
    """
    Cluster-Coupled Adam optimizer.
    - For param group:
      * K-means로 row 단위 클러스터링
      * 클러스터별 평균 gradient를 계산 후 mixing
      * Mixed gradient로 Adam 업데이트
    - param group 옵션:
      * clustered: True/False
      * num_clusters: K (elbow method, 2~K 중 선택)
      * alpha: 클러스터 공유 강도 (0~1)
      * recluster_interval: reclustering frequency (step)
      * cluster_source: "param" / "grad" / "ema_grad"
        * cluster_beta: EMA 계수 (cluster_source="ema_grad"일 때 사용)
      * cluster_start_step: 몇 번째 step부터 클러스터링을 켤지 (burn-in, default 0)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _find_best_k_elbow(self, w2d, k_min=2, k_max=16):
        Ks = list(range(k_min, k_max + 1))
        inertias = []

        for K in Ks:
            kmeans = KMeans(n_clusters=K, random_state=42)
            kmeans.fit(w2d)
            inertias.append(kmeans.inertia_)

        x1, y1 = Ks[0], inertias[0]
        x2, y2 = Ks[-1], inertias[-1]

        distances = []
        for x, y in zip(Ks, inertias):
            num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
            den = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            distances.append(num / (den + 1e-12))

        best_idx = int(np.argmax(distances))
        best_k = Ks[best_idx]
        return best_k

    @torch.no_grad()
    def _update_clusters(self, base_tensor, state, num_clusters, recluster_interval):
        """
        하나의 파라미터 텐서에 대해 K-means 클러스터링을 업데이트하는 함수.

        - base_tensor: 클러스터링 기준 텐서 (예: p.data, grad, EMA-grad)
        - state: 이 파라미터에 해당하는 optimizer state dict
        - num_clusters: elbow 탐색 시 사용할 최대 K (k_max 역할)
        - recluster_interval: 몇 step마다 클러스터링을 다시 할지
        """
        step = state.get("cluster_step", 0)
        assignments = state.get("assignments", None)

        # 아직 클러스터링 안 했거나, 일정 step마다 재클러스터링할 때만 새로 계산
        if assignments is None or step % recluster_interval == 0:
            w2d = base_tensor.detach().view(base_tensor.shape[0], -1).cpu().numpy()

            k_max = min(num_clusters, w2d.shape[0])

            if k_max < 2:
                assignments = torch.zeros(
                    w2d.shape[0], dtype=torch.long, device=base_tensor.device
                )
            else:
                best_K = state.get("best_K", None)
                if best_K is None:
                    best_K = self._find_best_k_elbow(w2d, k_min=2, k_max=k_max)
                    state["best_K"] = best_K

                    print(
                        f"[ClusterCoupledAdam] step={step} | "
                        f"rows={w2d.shape[0]} | k_max={k_max} | best_K(elbow)={best_K}"
                    )
                else:
                    best_K = min(best_K, k_max)

                if best_K < 2:
                    assignments = torch.zeros(
                        w2d.shape[0], dtype=torch.long, device=base_tensor.device
                    )
                else:
                    kmeans = KMeans(n_clusters=best_K, random_state=42)
                    labels = kmeans.fit_predict(w2d)
                    assignments = torch.from_numpy(labels).long().to(base_tensor.device)

            state["assignments"] = assignments

        state["cluster_step"] = step + 1

        return state["assignments"]

    @torch.no_grad()
    def step(self, closure=None):
        """
        - 각 param group과 파라미터에 대해:
          1) weight decay 적용 (Adam 방식)
          2) (clustered=True & burn-in 지나면) gradient를 클러스터 평균과 섞어서 g̃ 계산
          3) Adam 모멘트 업데이트 및 파라미터 갱신
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # cluster 관련 옵션
            clustered = group.get("clustered", False)
            alpha = group.get("alpha", 0.5)
            num_clusters = group.get("num_clusters", 8)
            recluster_interval = group.get("recluster_interval", 100)

            cluster_source = group.get("cluster_source", "param")   # "param" / "grad" / "ema_grad"
            cluster_beta   = group.get("cluster_beta", 0.9)         # EMA 계수 (0.9 ~ 0.99 정도)
            cluster_start_step = group.get("cluster_start_step", 0) # burn-in 끝나는 step

            # 개선 실험 옵션 (2026-10, 기본값이면 기존과 완전히 같은 동작)
            #   mix_mode = "grad"    : 기존. g̃ = (1-α)g + α·ḡ_C
            #              "dirnorm" : 크기를 맞춘 평균. ḡ 대신 ‖g_i‖·mean(g_j/‖g_j‖) (인기 행이 평균을 끄는 것 방지)
            #              "vshare"  : gradient는 그대로, 2차 모멘트만 클러스터 평균 쪽으로 섞음 (클러스터 단위 Coupled Adam)
            #   alpha_n0 > 0 : 행별 α_i = α·n0/(n0+n_i), n_i = 행의 학습 상호작용 수 (CEL: 드문 행일수록 공유)
            #   alpha_freq_dir = "warm" 이면 반대로 α_i = α·n_i/(n0+n_i) (인기 행일수록 공유)
            mix_mode = group.get("mix_mode", "grad")
            alpha_n0 = float(group.get("alpha_n0", 0.0) or 0.0)
            row_counts_list = group.get("row_counts", None)

            for p_idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("ClusterCoupledAdam does not support sparse gradients.")

                state = self.state[p]
                # state 초기화
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if clustered:
                        state["cluster_step"] = 0
                        state["assignments"] = None
                        if cluster_source == "ema_grad":
                            state["ema_grad"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                current_step = state["step"]

                # weight decay
                if weight_decay != 0.0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # shrink 모드: 행별 잡음 추정을 위해 처음부터 raw gradient의 1·2차 이동평균을 따로 기록
                if clustered and mix_mode == "shrink":
                    if "eb_m" not in state:
                        state["eb_m"] = torch.zeros_like(grad)
                        state["eb_s"] = torch.zeros_like(grad)
                    state["eb_m"].mul_(0.98).add_(grad, alpha=0.02)
                    state["eb_s"].mul_(0.98).addcmul_(grad, grad, value=0.02)

                # gradient mixing
                if clustered and current_step >= cluster_start_step:
                    if p.data.dim() != 2:
                        raise ValueError(
                            "Clustered param must be a 2D tensor (e.g., [num_embeddings, dim])."
                        )

                    # EMA-grad update
                    if cluster_source == "ema_grad":
                        ema = state.get("ema_grad", None)
                        if ema is None:
                            ema = torch.zeros_like(grad)
                            state["ema_grad"] = ema
                        ema.mul_(cluster_beta).add_(grad, alpha=1.0 - cluster_beta)
                        base_tensor = ema
                    elif cluster_source == "param":
                        base_tensor = p.data
                    elif cluster_source == "grad":
                        base_tensor = grad
                    else:
                        raise ValueError(f"Unknown cluster_source: {cluster_source}")

                    # clustering
                    assignments = self._update_clusters(
                        base_tensor, state, num_clusters, recluster_interval
                    )

                    if assignments.dim() != 1 or assignments.size(0) != p.shape[0]:
                        raise ValueError("Cluster assignments must be 1D (num_rows,).")

                    g2d = grad.view(p.shape[0], -1)
                    K = int(assignments.max().item() + 1)

                    # [수정 1] Active Mask 생성 (현재 배치에서 gradient가 발생한 행만 1, 나머진 0)
                    # 엡실론보다 큰 값을 가진 경우를 active로 간주
                    active_mask = (g2d.abs().sum(dim=1, keepdim=True) > 1e-10).float()
                    
                    # [수정 2] 분자(Sum) 계산: g2d에는 이미 0이 많으므로 그대로 더해도 됨
                    gc_sum = torch.zeros(K, g2d.size(1), device=grad.device)
                    idx_expanded = assignments.view(-1, 1).expand(-1, g2d.size(1))
                    gc_sum.scatter_add_(0, idx_expanded, g2d)

                    # [수정 3] 분모(Count) 계산: 전체 인원이 아니라, 'Active한 인원'만 카운트해야 함
                    gc_count = torch.zeros(K, 1, device=grad.device)
                    # active_mask가 1인 곳의 assignment 인덱스만 카운트
                    # active_mask가 1인 곳의 assignment를 가져와서 bincount 하거나 scatter_add 사용
                    # 여기서는 scatter_add 사용이 차원 맞추기 용이
                    active_assignments_idx = assignments.view(-1, 1)
                    gc_count.scatter_add_(0, active_assignments_idx, active_mask)
                    
                    gc_count = gc_count.clamp_min(1.0) # 0으로 나누기 방지
                    gc_avg = gc_sum / gc_count # 진정한 Active Average

                    # [수정 4] Mixing: Active한 행에 대해서만 Mixing 수행
                    # Active하지 않은 행(grad=0)은 그대로 0 유지 (mixed 연산에서 제외하거나 마스킹)
                    
                    # gc_avg[assignments]는 (N, dim) 형태
                    target_gc = gc_avg[assignments]
                    
                    # 행별 α (alpha_n0 > 0이면 빈도에 따라, 아니면 모든 행에 같은 α)
                    if alpha_n0 > 0 and row_counts_list is not None:
                        n_i = row_counts_list[p_idx].to(grad.device).view(-1, 1).float()
                        if group.get("alpha_freq_dir", "rare") == "warm":
                            # 인기 행일수록 많이 공유 (vshare와 함께: 드문 행은 자기 보폭 유지)
                            a = alpha * n_i / (alpha_n0 + n_i)
                        else:
                            a = alpha * alpha_n0 / (alpha_n0 + n_i)
                    else:
                        a = alpha

                    if mix_mode == "shrink":
                        # 경험적 베이즈(James–Stein형) 수축: α_i = σ_i² / (σ_i² + τ_C²)
                        #   σ_i² : 행 i gradient의 스텝 간 잡음 = mean_d(E[g²] − E[g]²)
                        #   τ_C² : 클러스터 안에서 행들의 평균 gradient가 서로 다른 정도 = mean_{j∈C} mean_d(E[g_j] − Ē_C)²
                        #   α(인자)는 상한으로 곱한다.
                        em = state["eb_m"].view(p.shape[0], -1)
                        es = state["eb_s"].view(p.shape[0], -1)
                        # Adam은 모멘텀(β1 EMA)으로 이미 잡음을 평균내므로, 한 스텝 잡음 σ² 대신
                        # EMA 추정값의 잡음 σ²·(1-β1)/(1+β1)을 쓴다 (그대로 쓰면 α≈1로 과하게 수축됨).
                        noise = (es - em * em).clamp_min(0).mean(dim=1, keepdim=True) * (1 - beta1) / (1 + beta1)
                        cnt_all = torch.bincount(assignments, minlength=K).clamp_min(1).view(-1, 1).float()
                        m_sum = torch.zeros(K, em.size(1), device=grad.device)
                        m_sum.scatter_add_(0, idx_expanded, em)
                        spread = ((em - (m_sum / cnt_all)[assignments]) ** 2).mean(dim=1, keepdim=True)
                        tau2 = torch.zeros(K, 1, device=grad.device)
                        tau2.scatter_add_(0, assignments.view(-1, 1), spread)
                        tau2 = (tau2 / cnt_all)[assignments]
                        a = a * noise / (noise + tau2 + 1e-12)
                        if current_step % 1000 == 0:
                            act = active_mask.view(-1) > 0
                            q = a.view(-1)[act]
                            if q.numel():
                                print(f"[shrink] step={current_step} rows={p.shape[0]} alpha mean={q.mean().item():.3f} "
                                      f"p10={q.quantile(0.1).item():.3f} p90={q.quantile(0.9).item():.3f}", flush=True)

                    if mix_mode == "dirnorm":
                        norms = g2d.norm(dim=1, keepdim=True)
                        unit = g2d / norms.clamp_min(1e-12) * active_mask
                        u_sum = torch.zeros(K, g2d.size(1), device=grad.device)
                        u_sum.scatter_add_(0, idx_expanded, unit)
                        target_gc = norms * (u_sum / gc_count)[assignments]

                    if mix_mode == "vshare":
                        # gradient는 섞지 않는다 (2차 모멘트는 아래 Adam 단계에서 섞음)
                        grad = (g2d * active_mask).view_as(p.grad)
                        state["vshare"] = (assignments, K, a)
                    else:
                        # Active한 녀석들만: (1-a)g + a*gc
                        # Inactive한 녀석들: 0
                        mixed = (1.0 - a) * g2d + a * target_gc

                        # 최종 마스킹: 원래 grad가 0이었던 곳은 업데이트 하지 않음!
                        grad = mixed * active_mask
                        grad = grad.view_as(p.grad)
                # else: burn-in 구간에서는 grad를 그대로 사용 (클러스터링 안 함)

                # 3) Adam 업데이트 (표준 Adam)
                beta1_t, beta2_t = beta1, beta2
                exp_avg.mul_(beta1_t).add_(grad, alpha=1 - beta1_t)
                exp_avg_sq.mul_(beta2_t).addcmul_(grad, grad, value=1 - beta2_t)

                vs = state.pop("vshare", None)
                if vs is not None:
                    # 클러스터 단위 Coupled Adam: v_eff = (1-a)·v_i + a·mean_{j∈C(i)} v_j
                    v_asg, v_K, v_a = vs
                    v2d = exp_avg_sq.view(p.shape[0], -1)
                    v_sum = torch.zeros(v_K, v2d.size(1), device=grad.device)
                    v_sum.scatter_add_(0, v_asg.view(-1, 1).expand(-1, v2d.size(1)), v2d)
                    v_cnt = torch.bincount(v_asg, minlength=v_K).clamp_min(1).view(-1, 1).float()
                    v_eff = (1.0 - v_a) * v2d + v_a * (v_sum / v_cnt)[v_asg]
                    denom = v_eff.view_as(exp_avg_sq).sqrt().add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)

                bias_correction1 = 1 - beta1_t ** current_step
                bias_correction2 = 1 - beta2_t ** current_step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

class ClusterCoupledSGD(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0.0,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _update_clusters(self, p, state, num_clusters, recluster_interval):
        step = state.get("cluster_step", 0)
        assignments = state.get("assignments", None)

        if assignments is None or step % recluster_interval == 0:
            w2d = p.data.detach().view(p.shape[0], -1).cpu().numpy()

            K = min(num_clusters, w2d.shape[0])
            if K < 2:
                assignments = torch.zeros(w2d.shape[0], dtype=torch.long, device=p.device)
            else:
                km = KMeans(n_clusters=K, random_state=42)
                labels = km.fit_predict(w2d)
                assignments = torch.from_numpy(labels).long().to(p.device)

            state["assignments"] = assignments

        state["cluster_step"] = step + 1
        return state["assignments"]

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            clustered = group.get("clustered", False)
            alpha = group.get("alpha", 0.5)
            num_clusters = group.get("num_clusters", 8)
            recluster_interval = group.get("recluster_interval", 100)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                    if clustered:
                        state["cluster_step"] = 0
                        state["assignments"] = None

                # weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # gradient mixing
                if clustered:
                    if p.data.dim() != 2:
                        raise ValueError("Clustered param must be 2D")

                    assignments = self._update_clusters(
                        p, state, num_clusters, recluster_interval
                    )

                    g2d = grad.view(assignments.numel(), -1)
                    K = int(assignments.max().item()) + 1

                    gc = torch.zeros(K, g2d.size(1), device=grad.device)
                    idx_expanded = assignments.view(-1, 1).expand(-1, g2d.size(1))
                    gc.scatter_add_(0, idx_expanded, g2d)

                    counts = torch.bincount(assignments, minlength=K).float().to(p.device)
                    counts = counts.clamp_min(1).unsqueeze(1)
                    gc = gc / counts

                    mixed = (1 - alpha) * g2d + alpha * gc[assignments]
                    grad = mixed.view_as(grad)

                # SGD with momentum
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                p.data.add_(buf, alpha=-lr)

        return loss

class ClusterCoupledRMSProp(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        alpha_cluster=0.5,
        num_clusters=16,
        recluster_interval=100,
        rho=0.9,
        eps=1e-8,
        weight_decay=0.0,
    ):
        defaults = dict(
            lr=lr,
            alpha_cluster=alpha_cluster,
            num_clusters=num_clusters,
            recluster_interval=recluster_interval,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def _update_clusters(self, p, state, num_clusters, recluster_interval):
        step = state.get("cluster_step", 0)
        assignments = state.get("assignments", None)

        if assignments is None or step % recluster_interval == 0:
            w2d = p.data.detach().view(p.shape[0], -1).cpu().numpy()

            K = min(num_clusters, w2d.shape[0])
            if K < 2:
                assignments = torch.zeros(w2d.shape[0], dtype=torch.long, device=p.device)
            else:
                km = KMeans(n_clusters=K, random_state=42)
                labels = km.fit_predict(w2d)
                assignments = torch.from_numpy(labels).long().to(p.device)

            state["assignments"] = assignments

        state["cluster_step"] = step + 1
        return state["assignments"]

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            rho = group["rho"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            alpha_c = group["alpha_cluster"]
            num_clusters = group["num_clusters"]
            recluster_interval = group["recluster_interval"]

            clustered = group.get("clustered", False)

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["square_avg"] = torch.zeros_like(p)
                    if clustered:
                        state["cluster_step"] = 0
                        state["assignments"] = None

                # weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # cluster mixing
                if clustered:
                    if p.data.dim() != 2:
                        raise ValueError("Clustered param must be 2D")

                    assignments = self._update_clusters(
                        p, state, num_clusters, recluster_interval
                    )

                    g2d = grad.view(assignments.numel(), -1)

                    K = int(assignments.max().item()) + 1
                    gc = torch.zeros(K, g2d.size(1), device=p.device)

                    idx_expanded = assignments.view(-1, 1).expand(-1, g2d.size(1))
                    gc.scatter_add_(0, idx_expanded, g2d)

                    counts = torch.bincount(assignments, minlength=K).float().to(p.device)
                    counts = counts.clamp_min(1).unsqueeze(1)

                    gc = gc / counts
                    mixed = (1 - alpha_c) * g2d + alpha_c * gc[assignments]
                    grad = mixed.view_as(grad)

                # update
                square_avg = state["square_avg"]
                square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)

                avg = grad / (square_avg.sqrt() + eps)

                p.data.add_(avg, alpha=-lr)

        return loss

class Lamb(Optimizer):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0,
                 clamp_value=10,
                 debias=True):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            clamp_value=clamp_value,
            debias=debias
        )
        super(Lamb, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            clamp_value = group['clamp_value']
            debias = group['debias']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients.")

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                step = state["step"]

                # Adam moment update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Adam bias correction
                if debias:
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    m = exp_avg / bias_correction1
                    v = exp_avg_sq / bias_correction2
                else:
                    m, v = exp_avg, exp_avg_sq

                adam_step = m / (v.sqrt() + eps)

                # Decoupled weight decay
                if wd != 0:
                    adam_step = adam_step + wd * p

                # Trust ratio
                w_norm = p.norm(p=2)
                g_norm = adam_step.norm(p=2)

                if w_norm.item() > 0 and g_norm.item() > 0:
                    trust_ratio = w_norm / g_norm
                else:
                    trust_ratio = 1.0

                trust_ratio = min(trust_ratio.item(), clamp_value)

                p.add_(adam_step, alpha=-lr * trust_ratio) # Parameter update

        return loss
