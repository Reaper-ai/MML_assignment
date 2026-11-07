class Spring:
    def __init__(self, k=20.0, d=10.0, m=1.0, x0=0.0):
        self.k = k
        self.damping_coefficient = d
        self.mass = m
        self.x0 = x0  


    def force(self, x):
        """Calculate the force exerted by the spring at position x."""
        return -self.k * (x - self.x0)

    def damping_force(self, velocity):
        """Calculate the damping force based on the velocity."""
        return -self.damping_coefficient * velocity

    def update(self, x, velocity, dt):
        """Update the position and velocity of the spring based on the forces acting on it."""
        # Calculate the net force
        spring_force = self.force(x)
        damping_force = self.damping_force(velocity)
        net_force = spring_force + damping_force        
        # Calculate acceleration using Newton's second law: F = m * a => a = F / m
        acceleration = net_force / self.mass
        
        # Update velocity and position using simple Euler integration
        new_velocity = velocity + acceleration * dt
        new_position = x + new_velocity * dt
        
        return new_position, new_velocity