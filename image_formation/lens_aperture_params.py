import numpy as np

import matplotlib.pyplot as plt


def thin_lens_zi(f, z_o):
    return 1 / (1 / f - 1 / z_o)

def get_aperture_diameter(f, f_number):
    return f / f_number

focal_lengths= [ 3,  9, 50, 200]
colors =       ['r','g','b','m']

# Create a plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for i in range(4):
    object_distances = np.linspace(1.1*focal_lengths[i], 10**4, int((10**4 - 1.1*focal_lengths[i])*4))
    image_distances = thin_lens_zi(focal_lengths[i], object_distances)
    
    ax1.loglog(object_distances, image_distances, color=colors[i], label=f"f = {focal_lengths[i]} mm")
    ax1.axvline(x=focal_lengths[i], color = colors[i], linestyle='--', alpha=0.5)

ax1.set_xlabel('Object Distance (z_o) [mm]')
ax1.set_ylabel('Image Distance (z_i) [mm]')
ax1.set_title('Lens-to-Image Distance vs Object Distance for Different Focal Lengths')
ax1.legend()
ax1.set_ylim(0, 3000)  



focal_lengths = [24,  50,   70,   200,  400 ,  600]
f_numbers =     [1.4, 1.8,  2.8,  2.8,  2.8,    4 ]
colors =        ['r', 'g',  'b',  'm',  'c',   'y']
for i in range(6):
    aperture_diameter = get_aperture_diameter(focal_lengths[i], f_numbers[i])
    
    ax2.scatter(focal_lengths[i], aperture_diameter, color=colors[i], label=f"f = {focal_lengths[i]} mm, N = {f_numbers[i]}")
    ax2.text(focal_lengths[i], aperture_diameter, f"  f#={f_numbers[i]} D={int(aperture_diameter)}", verticalalignment='bottom', horizontalalignment='left', color=colors[i])


plt.show()