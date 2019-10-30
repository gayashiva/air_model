from configparser import ConfigParser

config = ConfigParser()

config['settings'] = {
    'Input the Field Site Name: ' : 'schwarzsee',
    'Fountain discharge option(energy, temperature, schwarzsee): ' : 'schwarzsee',
}

config['icestupa'] = {
        'Fountain Water Temperature T_f' : '5',
        'Fountain flight time loss ftl' : '0.5',
        'Ice Emissivity ie' : '0.96',
        'Water emissivity we' : '0.95',
        'Density of Ice rho_i' : '916',
        'Cloudiness c' : '0.5',
        'Albedo of Ice a_i' : '0.6',
        'Albedo of Snow a_s' : '0.75',
        'Albedo of Water a_w' : '0.1',
        'Albedo decay rate t_d' : '21.9',
        'Density of Precipitation dp' : '70',
        'z0mi' : '0.001',
        'z0ms' : '0.0015',
        'z0hi' : '0.0001',
        'Shape s' : 'Cone',
        'Fountain steps h_f' : '3',
        'Ice layer thickness dx' : '0.001',
}

with open('./src/data/dev.ini', 'w') as f:
    config.write(f)
