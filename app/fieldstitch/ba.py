"""Sparse bundle adjustment (Levenberg-Marquardt with Schur complement).

Parameters
  cameras : rotation R (cam-from-world) and center C, 6 dof per camera
  intrinsics per group : f, cx, cy, k1, k2, p1, p2, k3  (OpenCV Brown model, fx = fy)
  points  : X (3 dof)
Priors on camera centers (GNSS/RTK), weak priors on rotation and intrinsics.
"""
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .geo import rodrigues_batch, rotvec_batch

NI = 8  # intrinsics per group


def project(R, C, intr, cam_group, X, oc, op, return_parts=False):
    Xc = np.einsum("mij,mj->mi", R[oc], X[op] - C[oc])
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    x = Xc[:, 0] / zs
    y = Xc[:, 1] / zs
    k = intr[cam_group[oc]]
    f, cx, cy, k1, k2, p1, p2, k3 = [k[:, i] for i in range(NI)]
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    rad = 1 + k1 * r2 + k2 * r4 + k3 * r6
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    uv = np.stack([f * xd + cx, f * yd + cy], 1)
    if return_parts:
        return uv, dict(Xc=Xc, z=zs, x=x, y=y, r2=r2, r4=r4, r6=r6, rad=rad, xd=xd, yd=yd,
                        f=f, k1=k1, k2=k2, k3=k3, p1=p1, p2=p2)
    return uv, z


def _skew(v):
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., 0, 1] = -v[..., 2]
    S[..., 0, 2] = v[..., 1]
    S[..., 1, 0] = v[..., 2]
    S[..., 1, 2] = -v[..., 0]
    S[..., 2, 0] = -v[..., 1]
    S[..., 2, 1] = v[..., 0]
    return S


def _bc(idx, w, n):
    return np.bincount(idx, weights=w, minlength=n)


class BundleAdjuster:
    def __init__(self, R, C, intr, cam_group, X, oc, op, uv, *, sigma_px=1.0,
                 C_prior=None, C_sigma=None, R_prior=None, rot_sigma_deg=10.0,
                 intr_prior=None, intr_sigma=None, optimize_intrinsics=True,
                 loss_scale=3.0, log=None):
        self.R = np.array(R, float)
        self.C = np.array(C, float)
        self.intr = np.array(intr, float)
        self.cg = np.asarray(cam_group, np.int64)
        self.X = np.array(X, float)
        self.oc = np.asarray(oc, np.int64)
        self.op = np.asarray(op, np.int64)
        self.uv = np.asarray(uv, float)
        self.sigma = float(sigma_px)
        self.N = self.R.shape[0]
        self.P = self.X.shape[0]
        self.G = self.intr.shape[0]
        self.Cp = None if C_prior is None else np.asarray(C_prior, float)
        self.Cs = None if C_sigma is None else np.asarray(C_sigma, float)
        self.Rp = None if R_prior is None else np.asarray(R_prior, float)
        self.rs = np.radians(rot_sigma_deg)
        self.Ip = self.intr.copy() if intr_prior is None else np.asarray(intr_prior, float)
        if intr_sigma is None:
            intr_sigma = np.array([0.02, 0.01, 0.01, 0.1, 0.1, 0.01, 0.01, 0.1])
            intr_sigma = np.tile(intr_sigma, (self.G, 1))
            intr_sigma[:, 0] *= self.intr[:, 0]
            intr_sigma[:, 1] *= self.intr[:, 1] * 2
            intr_sigma[:, 2] *= self.intr[:, 2] * 2
        self.Is = np.asarray(intr_sigma, float)
        self.opt_k = bool(optimize_intrinsics) and self.G > 0
        self.c = float(loss_scale)
        self.log = log or (lambda *a: None)
        # cameras / points that actually have observations
        self.cam_used = np.bincount(self.oc, minlength=self.N) > 0
        self._prepare_order()

    # ------------------------------------------------------------------ cost
    def _robust(self, s):
        c2 = self.c * self.c
        return c2 * np.log1p(s / c2), 1.0 / (1.0 + s / c2)

    def cost(self, R=None, C=None, intr=None, X=None):
        R = self.R if R is None else R
        C = self.C if C is None else C
        intr = self.intr if intr is None else intr
        X = self.X if X is None else X
        uv, z = project(R, C, intr, self.cg, X, self.oc, self.op)
        r = (uv - self.uv) / self.sigma
        s = (r * r).sum(1)
        s = np.where(z > 0, s, 1e6)
        rho, _ = self._robust(s)
        total = rho.sum()
        if self.Cp is not None:
            e = (C - self.Cp) / self.Cs
            total += (e * e).sum()
        if self.Rp is not None:
            e = rotvec_batch(np.einsum("nij,nkj->nik", R, self.Rp)) / self.rs
            total += (e * e).sum()
        if self.opt_k:
            e = (intr - self.Ip) / self.Is
            total += (e * e).sum()
        return total

    def residuals_px(self):
        uv, z = project(self.R, self.C, self.intr, self.cg, self.X, self.oc, self.op)
        e = np.linalg.norm(uv - self.uv, axis=1)
        e[z <= 0] = 1e6
        return e

    # ------------------------------------------------------------ linearize
    def _prepare_order(self):
        # observations sorted by camera; permutations for point / (group,point) ordering
        o = np.lexsort((self.op, self.oc))
        self.oc, self.op, self.uv = self.oc[o], self.op[o], self.uv[o]
        self.perm_p = np.argsort(self.op, kind="stable")
        self.op_s = self.op[self.perm_p]
        self.cam_ptr = np.r_[0, np.cumsum(np.bincount(self.oc, minlength=self.N))]
        self.pt_ptr = np.r_[0, np.cumsum(np.bincount(self.op, minlength=self.P))]
        if self.G > 0:
            key = self.cg[self.oc] * self.P + self.op
            self.perm_k = np.argsort(key, kind="stable")
            self.key_s = key[self.perm_k]

    @staticmethod
    def _seg_add(out, ids, arr):
        """out[ids] += arr for sorted ids (segment sums), ids sorted ascending."""
        if len(ids) == 0:
            return
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        out[ids[starts]] += np.add.reduceat(arr, starts, axis=0)

    def _blocks(self, ids, Aa, Bb, n, perm=None, chunk=200000):
        a, b = Aa.shape[2], Bb.shape[2]
        out = np.zeros((n, a, b))
        M = Aa.shape[0]
        for s0 in range(0, M, chunk):
            sl = slice(s0, min(M, s0 + chunk))
            if perm is None:
                A_, B_ = Aa[sl], Bb[sl]
            else:
                pp = perm[sl]
                A_, B_ = Aa[pp], Bb[pp]
            E = np.einsum("mki,mkj->mij", A_, B_)
            self._seg_add(out, ids[sl], E)
        return out

    def _vec(self, ids, Aa, r, n, perm=None):
        if perm is not None:
            Aa = Aa[perm]
            r = r[perm]
        E = np.einsum("mki,mk->mi", Aa, r)
        out = np.zeros((n, Aa.shape[2]))
        self._seg_add(out, ids, E)
        return out

    def _linearize(self):
        N, P, G = self.N, self.P, self.G
        oc, op = self.oc, self.op
        uv, d = project(self.R, self.C, self.intr, self.cg, self.X, oc, op, return_parts=True)
        r = (uv - self.uv) / self.sigma
        s = (r * r).sum(1)
        valid = d["z"] > 1e-6
        _, w = self._robust(s)
        w = np.where(valid, w, 0.0)
        sw = np.sqrt(w) / self.sigma
        r = r * np.sqrt(w)[:, None]

        x, y, z = d["x"], d["y"], d["z"]
        f = d["f"]
        k1, k2, k3, p1, p2 = d["k1"], d["k2"], d["k3"], d["p1"], d["p2"]
        r2, r4, r6, rad = d["r2"], d["r4"], d["r6"], d["rad"]
        dr = k1 + 2 * k2 * r2 + 3 * k3 * r4
        D00 = rad + 2 * x * x * dr + 2 * p1 * y + 6 * p2 * x
        D01 = 2 * x * y * dr + 2 * p1 * x + 2 * p2 * y
        D11 = rad + 2 * y * y * dr + 6 * p1 * y + 2 * p2 * x
        iz = 1.0 / z
        A = np.zeros((len(oc), 2, 3))
        a0 = f * sw * iz
        A[:, 0, 0] = a0 * D00
        A[:, 0, 1] = a0 * D01
        A[:, 0, 2] = -a0 * (D00 * x + D01 * y)
        A[:, 1, 0] = a0 * D01
        A[:, 1, 1] = a0 * D11
        A[:, 1, 2] = -a0 * (D01 * x + D11 * y)
        Rm = self.R[oc]
        JX = np.einsum("mij,mjk->mik", A, Rm)
        Jth = np.einsum("mij,mjk->mik", A, -_skew(d["Xc"]))
        Jc = np.concatenate([Jth, -JX], 2)  # (M,2,6)
        del A, Rm, Jth
        out = {}
        out["Ucc"] = Ucc = self._blocks(oc, Jc, Jc, N)
        out["gc"] = gc = self._vec(oc, Jc, r, N)
        out["V"] = self._blocks(self.op_s, JX, JX, P, perm=self.perm_p)
        out["gp"] = self._vec(self.op_s, JX, r, P, perm=self.perm_p)
        out["Wc"] = np.einsum("mki,mkj->mij", Jc, JX)  # (M,6,3), camera order
        if self.opt_k:
            Jk = np.zeros((len(oc), 2, NI))
            Jk[:, 0, 0] = d["xd"]
            Jk[:, 1, 0] = d["yd"]
            Jk[:, 0, 1] = 1
            Jk[:, 1, 2] = 1
            Jk[:, 0, 3] = f * x * r2
            Jk[:, 1, 3] = f * y * r2
            Jk[:, 0, 4] = f * x * r4
            Jk[:, 1, 4] = f * y * r4
            Jk[:, 0, 5] = f * 2 * x * y
            Jk[:, 1, 5] = f * (r2 + 2 * y * y)
            Jk[:, 0, 6] = f * (r2 + 2 * x * x)
            Jk[:, 1, 6] = f * 2 * x * y
            Jk[:, 0, 7] = f * x * r6
            Jk[:, 1, 7] = f * y * r6
            Jk *= sw[:, None, None]
            gidx = self.cg[oc]
            Ukk = np.zeros((G, NI, NI))
            gk = np.zeros((G, NI))
            for g in range(G):
                m = gidx == g
                Ukk[g] = np.einsum("mki,mkj->ij", Jk[m], Jk[m])
                gk[g] = np.einsum("mki,mk->i", Jk[m], r[m])
            out["Ukk"], out["gk"] = Ukk, gk
            out["Uck"] = self._blocks(oc, Jc, Jk, N)
            out["Wk"] = self._blocks(self.key_s, Jk, JX, G * P, perm=self.perm_k).reshape(G, P, NI, 3)
        # priors
        if self.Cp is not None:
            inv = 1.0 / self.Cs ** 2
            for a in range(3):
                Ucc[:, 3 + a, 3 + a] += inv[:, a]
            gc[:, 3:] += (self.C - self.Cp) * inv
        if self.Rp is not None:
            e = rotvec_batch(np.einsum("nij,nkj->nik", self.R, self.Rp))
            inv = 1.0 / self.rs ** 2
            for a in range(3):
                Ucc[:, a, a] += inv
            gc[:, :3] += e * inv
        if self.opt_k:
            inv = 1.0 / self.Is ** 2
            for a in range(NI):
                out["Ukk"][:, a, a] += inv[:, a]
            out["gk"] += (self.intr - self.Ip) * inv
        for a in range(6):
            Ucc[~self.cam_used, a, a] += 1.0
        return out

    # ---------------------------------------------------------------- solve
    def _solve(self, L, lam):
        N, P, G = self.N, self.P, self.G
        Ucc = L["Ucc"].copy()
        V = L["V"].copy()
        i6 = np.arange(6)
        i3 = np.arange(3)
        Ucc[:, i6, i6] += lam * np.maximum(Ucc[:, i6, i6], 1e-6)
        V[:, i3, i3] += lam * np.maximum(V[:, i3, i3], 1e-6) + 1e-9
        Vinv = np.linalg.inv(V)
        Wc = L["Wc"]
        Wb = sp.bsr_matrix((Wc, self.op, self.cam_ptr), shape=(6 * N, 3 * P))
        WTb = sp.bsr_matrix((Wc[self.perm_p].transpose(0, 2, 1), self.oc[self.perm_p], self.pt_ptr),
                            shape=(3 * P, 6 * N))
        Vb = sp.bsr_matrix((Vinv, np.arange(P), np.arange(P + 1)), shape=(3 * P, 3 * P))
        Ub = sp.bsr_matrix((Ucc, np.arange(N), np.arange(N + 1)), shape=(6 * N, 6 * N))
        Y = Wb @ Vb
        S = (Ub - Y @ WTb).tocsr()
        gp = L["gp"].ravel()
        rhs_c = -L["gc"].ravel() + Y @ gp
        Wk = None
        if self.opt_k:
            Ukk = L["Ukk"].copy()
            i8 = np.arange(NI)
            Ukk[:, i8, i8] += lam * np.maximum(Ukk[:, i8, i8], 1e-9)
            nk = NI * G
            Wk = np.zeros((nk, 3 * P))
            for g in range(G):
                Wk[g * NI:(g + 1) * NI] = L["Wk"][g].transpose(1, 0, 2).reshape(NI, 3 * P)
            VWkT = np.asarray(Vb @ Wk.T)
            Uck = np.zeros((6 * N, nk))
            for g in range(G):
                m = self.cg == g
                rows_g = (6 * np.where(m)[0][:, None] + i6[None]).ravel()
                Uck[rows_g, g * NI:(g + 1) * NI] = L["Uck"][m].reshape(-1, NI)
            Sck = Uck - np.asarray(Y @ Wk.T)
            Skk = np.zeros((nk, nk))
            for g in range(G):
                Skk[g * NI:(g + 1) * NI, g * NI:(g + 1) * NI] = Ukk[g]
            Skk -= Wk @ VWkT
            rhs_k = -L["gk"].ravel() + VWkT.T @ gp
            S = sp.bmat([[S, sp.csr_matrix(Sck)], [sp.csr_matrix(Sck.T), sp.csr_matrix(Skk)]])
            rhs = np.concatenate([rhs_c, rhs_k])
        else:
            rhs = rhs_c
        S = ((S + S.T) * 0.5).tocsr()
        dc = self._linsolve(S, rhs, N)
        dcam = dc[:6 * N].reshape(N, 6)
        t = gp + WTb @ dc[:6 * N]
        dk = None
        if self.opt_k:
            dk = dc[6 * N:].reshape(G, NI)
            t = t + Wk.T @ dc[6 * N:]
        dX = -np.einsum("pij,pj->pi", Vinv, t.reshape(P, 3))
        return dcam, dk, dX

    @staticmethod
    def _linsolve(S, rhs, N):
        n = S.shape[0]
        if n <= int(os.environ.get("FIELDSTITCH_DENSE_MAX", "4200")):
            try:
                import scipy.linalg as sla
                c = sla.cho_factor(S.toarray(), check_finite=False)
                x = sla.cho_solve(c, rhs, check_finite=False)
                if np.all(np.isfinite(x)):
                    return x
            except Exception:
                pass
        # block-Jacobi preconditioned conjugate gradients
        r6 = np.arange(N) * 6
        blocks = np.zeros((N, 6, 6))
        for a in range(6):
            for b in range(6):
                blocks[:, a, b] = np.asarray(S[r6 + a, r6 + b]).ravel()
        blocks[:, np.arange(6), np.arange(6)] += 1e-12
        binv = np.linalg.inv(blocks)
        nk = n - 6 * N
        kinv = np.linalg.pinv(S[6 * N:, 6 * N:].toarray()) if nk else None

        def prec(v):
            out = np.empty_like(v)
            out[:6 * N] = np.einsum("nij,nj->ni", binv, v[:6 * N].reshape(N, 6)).ravel()
            if nk:
                out[6 * N:] = kinv @ v[6 * N:]
            return out

        Mop = spla.LinearOperator((n, n), matvec=prec)
        try:
            x, _ = spla.cg(S, rhs, M=Mop, rtol=1e-7, maxiter=800)
        except TypeError:  # older scipy
            x, _ = spla.cg(S, rhs, M=Mop, tol=1e-7, maxiter=800)
        return x

    def _apply(self, dcam, dk, dX):
        R = rodrigues_batch(dcam[:, :3]) @ self.R
        C = self.C + dcam[:, 3:]
        intr = self.intr + dk if dk is not None else self.intr
        X = self.X + dX
        return R, C, intr, X

    def run(self, max_iter=30, tol=1e-5, lam=1e-3, time_limit=None):
        t0 = time.time()
        cost = self.cost()
        self.log("BA start: %d cams, %d pts, %d obs, cost %.4g" % (self.N, self.P, len(self.oc), cost))
        it = 0
        for it in range(max_iter):
            L = self._linearize()
            improved = False
            for _ in range(10):
                dcam, dk, dX = self._solve(L, lam)
                R, C, intr, X = self._apply(dcam, dk, dX)
                new = self.cost(R, C, intr, X)
                if np.isfinite(new) and new < cost:
                    self.R, self.C, self.intr, self.X = R, C, intr, X
                    rel = (cost - new) / max(cost, 1e-12)
                    cost = new
                    lam = max(lam / 3.0, 1e-8)
                    improved = True
                    break
                lam *= 5.0
                if lam > 1e10:
                    break
            if not improved:
                break
            if rel < tol:
                break
            if time_limit and time.time() - t0 > time_limit:
                self.log("BA time limit reached")
                break
        self.log("BA done: %d iterations, cost %.4g, %.1fs" % (it + 1, cost, time.time() - t0))
        return cost
