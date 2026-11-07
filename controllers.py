def Propotional(Kp, error):
    return Kp * error

def Integral(Ki, error, dt, integral):
    integral += error * dt
    return Ki * integral, integral

def Derivative(Kd, error, previous_error, dt):
    derivative = (error - previous_error) / dt
    return Kd * derivative

def PI(Kp, Ki, error, previous_error, dt, integral):
    p = Propotional(Kp, error)
    i, integral = Integral(Ki, error, dt, integral)
    return p + i, integral

def PD(Kp, Kd, error, previous_error, dt):
    p = Propotional(Kp, error)
    d = Derivative(Kd, error, previous_error, dt)
    return p + d

def PID(Kp, Ki, Kd, error, previous_error, dt, integral):
    p = Propotional(Kp, error)
    i, integral = Integral(Ki, error, dt, integral)
    d = Derivative(Kd, error, previous_error, dt)
    return p + i + d, integral


def Clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def PID_antiwindup(Kp, Ki, Kd, error, previous_error, dt, integral, min_value, max_value):
    """
    PID with conditional integration anti-windup.
    Only updates integral if output is not saturated in the direction of the error.
    """
    p = Propotional(Kp, error)
    d = Derivative(Kd, error, previous_error, dt)
    
    # Try adding the new integral
    integral_candidate = integral + error * dt
    i_candidate = Ki * integral_candidate
    
    u_raw = p + i_candidate + d
    u = Clamp(u_raw, min_value, max_value)
    
    # Conditional integration: only keep new integral if not pushing further into saturation
    if u == u_raw or (u_raw > max_value and error < 0) or (u_raw < min_value and error > 0):
        integral = integral_candidate
        i = i_candidate
    else:
        i = Ki * integral
    
    output = p + i + d
    output = Clamp(output, min_value, max_value)
    return output, integral


def PID_no_kick(Kp, Ki, Kd, error, pv, previous_pv, dt, integral):
    """PID with derivative on measurement (PV) - no kick on setpoint change."""
    p = Propotional(Kp, error)
    i, integral = Integral(Ki, error, dt, integral)
    d = -Kd * (pv - previous_pv) / dt
    return p + i + d, integral


def Derivative_with_filter(Kd, error, previous_error, dt, alpha, prev_filtered=0.0):
    derivative = (error - previous_error) / dt
    filtered_derivative = alpha * derivative + (1 - alpha) * prev_filtered
    return Kd * filtered_derivative, filtered_derivative


def Derivative_with_filter_pv(Kd, pv, previous_pv, dt, alpha, prev_filtered=0.0):
    derivative = (pv - previous_pv) / dt
    filtered_derivative = alpha * derivative + (1 - alpha) * prev_filtered
    return -Kd * filtered_derivative, filtered_derivative


def PID_filtered(Kp, Ki, Kd, error, pv, previous_pv, dt, integral, min_value, max_value, alpha=0.1,
                  prev_filtered=0.0):
    """PID with filtered derivative on PV and conditional anti-windup."""
    p = Propotional(Kp, error)
    
    integral_candidate = integral + error * dt
    i_candidate = Ki * integral_candidate
    
    d, new_filtered = Derivative_with_filter_pv(Kd, pv, previous_pv, dt, alpha, prev_filtered)
    
    u_raw = p + i_candidate + d
    u = Clamp(u_raw, min_value, max_value)
    
    if u == u_raw or (u_raw > max_value and error < 0) or (u_raw < min_value and error > 0):
        integral = integral_candidate
        i = i_candidate
    else:
        i = Ki * integral
    
    output = p + i + d
    output = Clamp(output, min_value, max_value)
    return output, integral, new_filtered