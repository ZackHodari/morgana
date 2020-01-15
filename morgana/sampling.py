import math

import torch
from torch.distributions import Distribution, Normal


class UniformSphereSurfaceSampler(Distribution):
    r"""Samples points uniformly on an n-dimensional sphere's surface.

    Notes
    -----
    This is the same sampling procedure used by the von Mises-Fisher distribution for :math:`\kappa = 0`.
    `von Mises-Fisher distribution <https://en.wikipedia.org/wiki/Von_Mises–Fisher_distribution>`_
    """
    def __init__(self, centre, radius):
        self.device = centre.device

        self.centre = centre
        self.dim = len(self.centre)
        self.radius = radius
        self.normal = Normal(0, 1)

        super(UniformSphereSurfaceSampler, self).__init__()

    def rsample(self, sample_shape=torch.Size()):
        r"""Samples points on the surface of the hypersphere."""
        direction = self.normal.rsample(list(sample_shape) + [self.dim]).to(self.device)
        point_on_unit_sphere = direction / torch.norm(direction, dim=-1, p=None, keepdim=True)
        return self.centre + self.radius * point_on_unit_sphere


def evenly_spaced_points_on_hypersphere(centre, radius, N):
    r"""We want the surface of an n-ball, which is equivalent to an (n-1)-sphere.

    S_n(R) = ( 2 * pi ^ ((n + 1) / 2) ) / Gamma((n/1) + 1) * R^n
    """
    dim = len(centre)

    points = []
    offset = 2. / N
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(N):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + 1.) % N) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return points



class UniformEllipsoidSurfaceApproximateSampler(Distribution):
    r"""Samples points ~uniformly on an n-dimensional ellipse's surface.

    Notes
    -----
    This is not a fair sampler, at the poles (especially for dimensions with large radii) samples will be denser.
    """
    def __init__(self, centre, radii):
        super(UniformEllipsoidSurfaceApproximateSampler, self).__init__()

        self.centre = centre
        self.radii = radii

        self.phi = torch.distributions.Uniform(low=0., high=2 * math.pi)
        self.theta = torch.distributions.Uniform(low=0., high=math.pi)

        self.ndims = centre.shape[-1]

    def sample_angles(self, sample_shape):
        r"""Samples angles from n-1 uniform distributions.

        One of these angles is in the range [0, 2*pi] and it determines tha angle in the first two dimensions.
        The remaining n-2 angles are in the range [0, pi] and determine the angle in the remaining dimensions.
        """
        phi = self.phi.rsample(sample_shape + [1])
        thetas = self.theta.rsample(sample_shape + [max(0, self.ndims - 2)])

        return torch.cat((phi, thetas), dim=-1)

    def rsample(self, sample_shape=torch.Size()):
        r"""Computes the transformation for each cartesian dimension `n`.

        .. math::

            x_1     &= r_1     * \cos( \theta_1 )

            x_2     &= r_2     * \sin( \theta_1 ) * \cos( \theta_2 )

            x_3     &= r_3     * \sin( \theta_1 ) * \sin( \theta_2 ) * \cos( \theta_3 )

            ...

            x_{n-2} &= r_{n-2} * \sin( \theta_1 ) * \sin( \theta_2 ) * ... * \sin( \theta_{n-3} ) * \cos( \theta_{n-2} )

            x_{n-1} &= r_{n-1} * \sin( \theta_1 ) * \sin( \theta_2 ) * ... * \sin( \theta_{n-2} ) * \cos( \theta_{n-1} )

            x_n     &= r_{n}   * \sin( \theta_1 ) * \sin( \theta_2 ) * ... * \sin( \theta_{n-1} )


        Note the first dimension :math:`x_1` does not use :math:`\sin`, and the final dimension :math:`x_n` does not use
        :math:`\cos`.

        This is equivalent to,

        .. math::

            \mathtt{cumprod\_sin}_1 &= 1.

            \mathtt{cumprod\_sin}_n &= \prod_{i=1}^n \sin( \theta_i )

            \\

            \mathtt{cos}_n &= \cos( \theta_n )

            \mathtt{cos}_N &= 1.

            \\

            x_n &= r_n * \mathtt{cumprod\_sin}_n * \mathtt{cos}_n

        """
        angles = self.sample_angles(sample_shape)

        cumprod_sin = torch.cumprod(torch.sin(angles), dim=-1)
        cos = torch.cos(angles)

        pad = torch.ones_like(cumprod_sin[..., [0]])

        cumprod_sin = torch.cat((pad, cumprod_sin), dim=-1)
        cos_padded = torch.cat((cos, pad), dim=-1)

        return self.radii * cumprod_sin * cos_padded

