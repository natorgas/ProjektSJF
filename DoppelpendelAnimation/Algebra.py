import sympy as smp

# Differentialgleichungen werden in die Richtige Form für spätere Berechnungen gebracht

t, g, l1, l2, m1, m2, alpha, beta, alpha_dot, beta_dot, alpha_double_dot, beta_double_dot, v1, v2 = smp.symbols(r"t g l1 l2 m1 m2 alpha beta \dot{\alpha} \dot{\beta} \ddot{\alpha} \ddot{\beta} v1 v2")

expr1 = smp.Eq(l1**2*alpha_double_dot*(m1+m2)+m2*l1*l2*beta_double_dot*smp.cos(alpha-beta)+m2*l1*l2*beta_dot**2*smp.sin(alpha-beta)+g*l1*smp.sin(alpha)*(m1+m2), 0)
expr2 = smp.Eq(l2*beta_double_dot+l1*alpha_double_dot*smp.cos(alpha-beta)-l1*alpha_dot**2*smp.sin(alpha-beta)+g*smp.sin(beta), 0)

expr1_subs = expr1.subs(beta_double_dot, smp.solve(expr2, beta_double_dot)[0])
expr2_subs = expr2.subs(alpha_double_dot, smp.solve(expr1, alpha_double_dot)[0])

expr_for_alpha_dd = smp.solve(expr1_subs, alpha_double_dot)[0].subs(alpha_dot, v1).subs(beta_dot, v2)
expr_for_beta_dd = smp.solve(expr2_subs, beta_double_dot)[0].subs(alpha_dot, v1).subs(beta_dot, v2)

v1_dot = smp.lambdify([g, m1, m2, alpha, beta, l1, l2, v1, v2], expr_for_alpha_dd)
v2_dot = smp.lambdify([g, m1, m2, alpha, beta, l1, l2, v1, v2], expr_for_beta_dd)


