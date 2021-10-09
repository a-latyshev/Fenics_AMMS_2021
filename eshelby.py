import dolfin

import numpy as np


class EshelbyDisk:
    def __init__(self, γ, χ, ν_i, ν_m):
        self.γ = γ
        self.χ = χ
        self.ν_i = ν_i
        self.ν_m = ν_m

    def create_lhs(self):
        return np.array(
            [
                [2, 2 * self.γ ** 2, 2 * self.γ ** 4, 2 * self.γ ** 6, 0, 0],
                [
                    2 * self.ν_m * (self.ν_m - 1),
                    -(self.γ ** 2) * self.ν_m * (2 * self.ν_m - 1),
                    -2 * self.γ ** 4 * self.ν_m * (self.ν_m - 1),
                    self.γ ** 6 * (self.ν_m - 1) * (2 * self.ν_m - 3),
                    0,
                    0,
                ],
                [1, 1, 1, 1, -1, -1],
                [
                    2 * self.ν_i * self.ν_m * (self.ν_m - 1),
                    -self.ν_i * self.ν_m * (2 * self.ν_m - 1),
                    -2 * self.ν_i * self.ν_m * (self.ν_m - 1),
                    self.ν_i * (self.ν_m - 1) * (2 * self.ν_m - 3),
                    2 * self.ν_i * self.ν_m * (self.ν_m - 1),
                    -self.ν_m * (2 * self.ν_i - 3) * (self.ν_m - 1),
                ],
                [
                    3 * (self.ν_i + 1) * (self.ν_m - 1),
                    -self.ν_i - 1,
                    -(self.ν_i + 1) * (self.ν_m - 1),
                    0,
                    self.χ * (self.ν_m - 1) * (self.ν_m + 1),
                    0,
                ],
                [
                    -6 * self.ν_i * self.ν_m * (self.ν_i + 1) * (self.ν_m - 1),
                    self.ν_i * self.ν_m * (self.ν_i + 1),
                    -2 * self.ν_i * self.ν_m * (self.ν_i + 1) * (self.ν_m - 1),
                    -3 * self.ν_i * (self.ν_i + 1) * (self.ν_m - 1),
                    2 * self.χ * self.ν_i * self.ν_m * (self.ν_m - 1) * (self.ν_m + 1),
                    3 * self.χ * self.ν_m * (self.ν_m - 1) * (self.ν_m + 1),
                ],
            ]
        )

    def create_rhs(self):
        return np.array(
            [2 * self.γ ** 4, -2 * self.γ ** 4 * self.ν_m * (self.ν_m - 1), 0, 0, 0, 0]
        )

    def compute_integration_constants(self):
        return np.linalg.solve(self.create_lhs(), self.create_rhs())

    def to_expression(self, a=1.0, degree=4):
        A_m3, A_m1, A_1, A_3, C_1, C_3 = self.compute_integration_constants()
        ρ = dolfin.Expression("sqrt(pow(x[0], 2)+pow(x[1], 2))/a", degree=degree, a=a)
        θ = dolfin.Expression("atan2(x[1], x[0])", degree=degree)

        params = {
            "A_m3": A_m3,
            "A_m1": A_m1,
            "A_1": A_1,
            "A_3": A_3,
            "B_m3": -A_m3,
            "B_m1": (1 - 2 * self.ν_m) / 2 / (1 - self.ν_m) * A_m1,
            "B_1": A_1,
            "B_3": (3 - 2 * self.ν_m) / 2 / self.ν_m * A_3,
            "C_1": C_1,
            "C_3": C_3,
            "D_1": C_1,
            "D_3": (3 - 2 * self.ν_i) / 2 / self.ν_i * C_3,
            "rho": ρ,
            "theta": θ,
        }

        F = dolfin.Expression(
            "rho > 1 ? A_m3*pow(rho, -3)+A_m1/rho+A_1*rho+A_3*pow(rho, 3): C_1*rho+C_3*pow(rho, 3)",
            degree=degree,
            **params
        )
        G = dolfin.Expression(
            "rho > 1 ? B_m3*pow(rho, -3)+B_m1/rho+B_1*rho+B_3*pow(rho, 3): D_1*rho+D_3*pow(rho, 3)",
            degree=degree,
            **params
        )
        u_r = dolfin.Expression("-F*sin(2*theta)", degree=degree, F=F, rho=ρ, theta=θ) #seb2021 j ai mis le -
        u_θ = dolfin.Expression("-G*cos(2*theta)", degree=degree, G=G, rho=ρ, theta=θ) #seb2021 j ai mis le -
        u = dolfin.Expression(
            ("u_r*cos(theta)-u_theta*sin(theta)", "u_r*sin(theta)+u_theta*cos(theta)"),
            degree=degree,
            theta=θ,
            u_r=u_r,
            u_theta=u_θ,
        )
        return u
