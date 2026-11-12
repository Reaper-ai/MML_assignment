def P(Kp, error):
    """Proportional term."""
    return Kp * error


def I(Ki, error, dt, integral):
    """Pure integral term (no anti-windup)."""
    integral += error * dt
    return Ki * integral, integral


def D(Kd, error, previous_error, dt):
    """Derivative on error."""
    derivative = (error - previous_error) / dt
    return Kd * derivative


def D_filtered(Kd, error, previous_error, dt, alpha, prev_filtered=0.0):
    """Filtered derivative on error."""
    derivative = (error - previous_error) / dt
    filtered_derivative = alpha * derivative + (1 - alpha) * prev_filtered
    return Kd * filtered_derivative, filtered_derivative


def D_filtered_pv(Kd, pv, previous_pv, dt, alpha, prev_filtered=0.0):
    """Filtered derivative on process variable (no kick + noise rejection)."""
    derivative = (pv - previous_pv) / dt
    filtered_derivative = alpha * derivative + (1 - alpha) * prev_filtered
    return -Kd * filtered_derivative, filtered_derivative


def Clamp(value, min_val, max_val):
    """Output saturation (now used by the plant/physics sim, not controller)."""
    return max(min(value, max_val), min_val)

def I_anti_windup(Ki, error, dt, integral, min_val, max_val, u_raw):
    """Integral term with conditional anti-windup."""
    # True Anti-Windup: Pause integration if saturated AND error is pushing further into saturation
    if (u_raw > max_val and error > 0) or (u_raw < min_val and error < 0):
        new_integral = integral
    else:
        new_integral = integral + error * dt
        
    i_term = Ki * new_integral
    return i_term, new_integral


# ===== Composed Controllers =====

def PI(Kp, Ki, error, previous_error, dt, integral):
    """PI = P + I"""
    p = P(Kp, error)
    i, integral = I(Ki, error, dt, integral)
    return p + i, integral


def PD(Kp, Kd, error, previous_error, dt):
    """PD = P + D"""
    p = P(Kp, error)
    d = D(Kd, error, previous_error, dt)
    return p + d


def PID(Kp, Ki, Kd, error, previous_error, dt, integral):
    """Standard PID = P + I + D"""
    p = P(Kp, error)
    i, integral = I(Ki, error, dt, integral)
    d = D(Kd, error, previous_error, dt)
    return p + i + d, integral


def PID_no_windup(Kp, Ki, Kd, error, previous_error, dt, integral, min_val, max_val):
    """PID with conditional anti-windup."""
    p = P(Kp, error)
    d = D(Kd, error, previous_error, dt)
    i_term, new_integral = I_anti_windup(Ki, error, dt, integral, min_val, max_val, p + Ki * integral + d)

    u_requested = p + i_term + d
    
    # Return requested theoretical output (Plant handles actual clamping)
    return u_requested, new_integral


def PID_filtered(Kp, Ki, Kd, error, previous_error, dt, integral, alpha=0.1, prev_filtered=0.0):
    """PID with filtered derivative on ERROR = P + I + D_filtered"""
    p = P(Kp, error)
    i, integral = I(Ki, error, dt, integral)
    d, new_filtered = D_filtered(Kd, error, previous_error, dt, alpha, prev_filtered)
    
    u_requested = p + i + d
    return u_requested, integral, new_filtered


def PID_no_kick(Kp, Ki, Kd, error, pv, previous_pv, dt, integral, alpha=0.1, prev_filtered=0.0):
    """PID with D on PV + filter (no kick + noise) = P + I + D_filtered_pv"""
    p = P(Kp, error)
    i, integral = I(Ki, error, dt, integral)
    d, new_filtered = D_filtered_pv(Kd, pv, previous_pv, dt, alpha, prev_filtered)
    
    u_requested = p + i + d
    return u_requested, integral, new_filtered


def PID_robust_all(Kp, Ki, Kd, error, pv, previous_pv, dt, integral, min_val, max_val, alpha=0.1, prev_filtered=0.0):
    """
    Fully robust PID = P + Anti-windup + D_filtered_pv
    """
    p = P(Kp, error)
    d, new_filtered = D_filtered_pv(Kd, pv, previous_pv, dt, alpha, prev_filtered)
    
    i_term, new_integral = I_anti_windup(Ki, error, dt, integral, min_val, max_val, p + Ki * integral + d)
    u_requested = p + i_term + d
    
    return u_requested, new_integral, new_filtered