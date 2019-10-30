site = input("Input the Field Site Name: ") or "schwarzsee"

option = (
    input("Fountain discharge option(energy, temperature, schwarzsee): ")
    or "schwarzsee"
)

fountain = dict(
    T_f=5,  # Fountain Water Temperature T_f
    ftl=0.5,  # Fountain flight time loss ftl
    d_f=0.005,  # Fountain hole diameter
    h_f=3,  # Fountain steps h_f
    theta_f=45,  # Fountain aperture angle
    s=0,  # Shape s
)

weather = dict(
    c=0.5,  # Cloudiness c
    theta_s=45,  # Solar Angle
    z0mi=0.001,  # Ice Momentum roughness length
    z0hi=0.0001,  # Ice Scalar roughness length
)

surface = dict(
    ie=0.96,  # Ice Emissivity ie
    we=0.95,  # Water emissivity we
    a_i=0.6,  # Albedo of Ice a_i
    a_s=0.75,  # Albedo of Snow a_s
    a_w=0.1,  # Albedo of Water a_w
    t_d=21.9,  # Albedo decay rate t_d
)
