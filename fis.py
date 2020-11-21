import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Generate universe variables
#   * HOG, ORB, LBP, and DOG on subjective ranges [0, 1]

#   * output is a booleanv value of [<50, >50] in units of percentage points
hog = np.arange(0, 1, 0.1)
orb = np.arange(0, 1, 0.1)
lbp = np.arange(0, 1, 0.1)
dog = np.arange(0, 1, 0.1)

# Generate fuzzy membership functions
# HOG
hog_nm = fuzz.trapmf(hog, [0.0, 0.0, 0.4, 0.5])
hog_z = fuzz.trapmf(hog, [0.4, 0.50, 0.65, 0.7])
hog_pm = fuzz.trapmf(hog, [0.65, 0.7, 1.0, 1.0])

# ORB
orb_nm = fuzz.trapmf(orb, [0.0, 0.0, 0.3, 0.5])
orb_z = fuzz.trapmf(orb, [0.4, 0.5, 0.6, 0.7])
orb_pm = fuzz.trapmf(orb, [0.65, 0.7, 1.0, 1.0])

# LBP
lbp_nm = fuzz.trapmf(lbp, [0.0, 0.0, 0.4, 0.5])
lbp_z = fuzz.trapmf(lbp, [0.4, 0.5, 0.65, 0.7])
lbp_pm = fuzz.trapmf(lbp, [0.65, 0.7, 1.0, 1.0])

# DOG
dog_nm = fuzz.trapmf(dog, [0.0, 0.0, 0.4, 0.5])
dog_z = fuzz.trapmf(dog, [0.5, 0.6, 0.7, 0.8])
dog_pm = fuzz.trapmf(dog, [0.7, 0.8, 1.0, 1.0])



# Visualize these universes and membership functions
# fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

# ax0.plot(hog, hog_pm, 'b', linewidth=1.5, label='High')
# ax0.plot(hog, hog_z, 'g', linewidth=1.5, label='Midium')
# ax0.plot(hog, hog_nm, 'r', linewidth=1.5, label='Low')
# ax0.set_title('HOG')
# ax0.legend()
#
# ax1.plot(sift, sift_pm, 'b', linewidth=1.5, label='High')
# ax1.plot(sift, sift_z, 'g', linewidth=1.5, label='Midium')
# ax1.plot(sift, sift_nm, 'r', linewidth=1.5, label='Low')
# ax1.set_title('SIFT')
# ax1.legend()

# ax2.plot(lbp, lbp_pm, 'b', linewidth=1.5, label='High')
# ax2.plot(lbp, lbp_z, 'g', linewidth=1.5, label='Midium')
# ax2.plot(lbp, lbp_nm, 'r', linewidth=1.5, label='Low')
# ax2.set_title('LBP')
# ax2.legend()
#
# ax3.plot(dog, dog_pm, 'b', linewidth=1.5, label='High')
# ax3.plot(dog, dog_z, 'g', linewidth=1.5, label='Medium')
# ax3.plot(dog, dog_nm, 'r', linewidth=1.5, label='Low')
# ax3.set_title('DOG')
# ax3.legend()
#
# Turn off top/right axes
# for ax in (ax0, ax1, ax2, ax3):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.get_xaxis().tick_bottom()
#     ax.get_yaxis().tick_left()

# plt.tight_layout()
# plt.savefig('membership_function2.png')

universe = np.arange(1, 100, 5)

# print(universe)

# Create the three fuzzy variables - two inputs, one output
hog_v = ctrl.Antecedent(hog, 'HOG')
orb_v = ctrl.Antecedent(orb, 'ORB')
lbp_v = ctrl.Antecedent(lbp, 'LBP')
dog_v = ctrl.Antecedent(dog, 'DOG')

output = ctrl.Consequent(universe, 'output')

# Here we use the convenience `automf` to populate the fuzzy variables with
# terms. The optional kwarg `names=` lets us specify the names of our Terms.
names = ['nm', 'ze', 'pm']
output_name = ['Generate', 'Rank 5','Rank 3']
hog_v.automf(names=names)
orb_v.automf(names=names)
dog_v.automf(names=names)
lbp_v.automf(names=names)
output.automf(names=output_name)

rule0 = ctrl.Rule(antecedent=((hog_v['nm'] & orb_v['nm']) &
                              (lbp_v['nm'] & dog_v['nm'])),
                              consequent=output['Generate'])


rule1 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['nm']) &
                               (lbp_v['nm']  & dog_v['ze'])),
                  consequent=output['Generate'])

rule2 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['nm']) &
                               (lbp_v['nm']  & dog_v['pm'])),
                  consequent=output['Rank 5'])

rule3 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['nm']) &
                               (lbp_v['ze']  & dog_v['nm'])),
                  consequent=output['Generate'])

rule4 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['nm']) &
                               (lbp_v['ze']  & dog_v['ze'])),
                  consequent=output['Rank 5'])

rule5 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['nm']) &
                               (lbp_v['ze']  & dog_v['pm'])),
                  consequent=output['Rank 5'])

rule6 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['nm']) &
                               (lbp_v['pm']  & dog_v['nm'])),
                  consequent=output['Generate'])

rule7 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['nm']) &
                               (lbp_v['pm']  & dog_v['ze'])),
                  consequent=output['Rank 5'])



rule8 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['nm']) &
                               (lbp_v['pm']  & dog_v['pm'])),
                  consequent=output['Rank 5'])


rule9 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['ze']) &
                               (lbp_v['nm']  & dog_v['nm'])),
                  consequent=output['Generate'])

rule10 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['ze']) &
                               (lbp_v['nm']  & dog_v['ze'])),
                  consequent=output['Rank 5'])


rule11 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['ze']) &
                               (lbp_v['nm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule12 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['ze']) &
                               (lbp_v['ze']  & dog_v['nm'])),
                  consequent=output['Rank 5'])

rule13 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['ze']) &
                               (lbp_v['ze']  & dog_v['ze'])),
                  consequent=output['Rank 5'])

rule14 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['ze']) &
                               (lbp_v['ze']  & dog_v['pm'])),
                  consequent=output['Rank 3'])



rule15 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['ze']) &
                               (lbp_v['pm']  & dog_v['nm'])),
                  consequent=output['Rank 5'])


rule16 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['ze']) &
                               (lbp_v['pm']  & dog_v['ze'])),
                  consequent=output['Rank 3'])

rule17 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['ze']) &
                               (lbp_v['pm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])


rule18 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['pm']) &
                               (lbp_v['nm']  & dog_v['nm'])),
                  consequent=output['Generate'])

rule19 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['pm']) &
                               (lbp_v['nm']  & dog_v['ze'])),
                  consequent=output['Rank 5'])

rule20 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['pm']) &
                               (lbp_v['nm']  & dog_v['pm'])),
                  consequent=output['Rank 5'])

rule21 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['pm']) &
                               (lbp_v['ze']  & dog_v['nm'])),
                  consequent=output['Rank 5'])

rule22 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['pm']) &
                               (lbp_v['ze']  & dog_v['ze'])),
                  consequent=output['Rank 5'])


rule23 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['pm']) &
                               (lbp_v['ze']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule24 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['pm']) &
                               (lbp_v['pm']  & dog_v['nm'])),
                  consequent=output['Rank 5'])


rule25 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['pm']) &
                               (lbp_v['pm']  & dog_v['ze'])),
                  consequent=output['Rank 3'])

rule26 = ctrl.Rule(antecedent = ((hog_v['nm'] & orb_v['pm']) &
                               (lbp_v['pm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule27 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['nm']) &
                               (lbp_v['nm']  & dog_v['nm'])),
                  consequent=output['Generate'])

rule28 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['nm']) &
                               (lbp_v['nm']  & dog_v['ze'])),
                  consequent=output['Generate'])

rule29 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['nm']) &
                               (lbp_v['nm']  & dog_v['pm'])),
                  consequent=output['Generate'])


rule30 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['nm']) &
                               (lbp_v['ze']  & dog_v['nm'])),
                  consequent=output['Generate'])

rule31 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['nm']) &
                               (lbp_v['ze']  & dog_v['ze'])),
                  consequent=output['Rank 5'])


rule32 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['nm']) &
                               (lbp_v['ze']  & dog_v['pm'])),
                  consequent=output['Rank 5'])

rule33 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['nm']) &
                               (lbp_v['pm']  & dog_v['nm'])),
                  consequent=output['Generate'])

rule34 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['nm']) &
                               (lbp_v['pm']  & dog_v['ze'])),
                  consequent=output['Rank 5'])

rule35 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['nm']) &
                               (lbp_v['pm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])



rule36 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['ze']) &
                               (lbp_v['nm']  & dog_v['nm'])),
                  consequent=output['Rank 5'])


rule37 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['ze']) &
                               (lbp_v['nm']  & dog_v['ze'])),
                  consequent=output['Rank 5'])

rule38 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['ze']) &
                               (lbp_v['nm']  & dog_v['pm'])),
                  consequent=output['Rank 5'])


rule39 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['ze']) &
                               (lbp_v['ze']  & dog_v['nm'])),
                  consequent=output['Rank 5'])

rule40 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['ze']) &
                               (lbp_v['ze']  & dog_v['ze'])),
                  consequent=output['Rank 5'])

rule41 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['ze']) &
                               (lbp_v['ze']  & dog_v['pm'])),
                  consequent=output['Rank 5'])

rule42 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['ze']) &
                               (lbp_v['pm']  & dog_v['nm'])),
                  consequent=output['Rank 5'])

rule43 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['ze']) &
                               (lbp_v['pm']  & dog_v['ze'])),
                  consequent=output['Rank 3'])


rule44 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['ze']) &
                               (lbp_v['pm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule45 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['pm']) &
                               (lbp_v['nm']  & dog_v['nm'])),
                  consequent=output['Rank 5'])


rule46 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['pm']) &
                               (lbp_v['nm']  & dog_v['ze'])),
                  consequent=output['Rank 5'])

rule47 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['pm']) &
                               (lbp_v['nm']  & dog_v['pm'])),
                  consequent=output['Rank 5'])

rule48 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['pm']) &
                               (lbp_v['ze']  & dog_v['nm'])),
                  consequent=output['Rank 3'])

rule49 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['pm']) &
                               (lbp_v['ze']  & dog_v['ze'])),
                  consequent=output['Rank 3'])

rule50 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['pm']) &
                               (lbp_v['ze']  & dog_v['pm'])),
                  consequent=output['Rank 3'])


rule51 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['pm']) &
                               (lbp_v['pm']  & dog_v['nm'])),
                  consequent=output['Rank 3'])

rule52 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['pm']) &
                               (lbp_v['pm']  & dog_v['ze'])),
                  consequent=output['Rank 3'])


rule53 = ctrl.Rule(antecedent = ((hog_v['ze'] & orb_v['pm']) &
                               (lbp_v['pm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule54 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['nm']) &
                               (lbp_v['nm']  & dog_v['nm'])),
                  consequent=output['Generate'])

rule55 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['nm']) &
                               (lbp_v['nm']  & dog_v['ze'])),
                  consequent=output['Generate'])

rule56 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['nm']) &
                               (lbp_v['nm']  & dog_v['pm'])),
                  consequent=output['Rank 5'])

rule57 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['nm']) &
                               (lbp_v['ze']  & dog_v['nm'])),
                  consequent=output['Rank 5'])


rule58 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['nm']) &
                               (lbp_v['ze']  & dog_v['ze'])),
                  consequent=output['Rank 5'])

rule59 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['nm']) &
                               (lbp_v['ze']  & dog_v['pm'])),
                  consequent=output['Rank 5'])


rule60 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['nm']) &
                               (lbp_v['pm']  & dog_v['nm'])),
                  consequent=output['Rank 5'])

rule61 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['nm']) &
                               (lbp_v['pm']  & dog_v['ze'])),
                  consequent=output['Rank 5'])

rule62 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['nm']) &
                               (lbp_v['pm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule63 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['ze']) &
                               (lbp_v['nm']  & dog_v['nm'])),
                  consequent=output['Rank 5'])

rule64 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['ze']) &
                               (lbp_v['nm']  & dog_v['ze'])),
                  consequent=output['Rank 5'])


rule65 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['ze']) &
                               (lbp_v['nm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule66 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['ze']) &
                               (lbp_v['ze']  & dog_v['nm'])),
                  consequent=output['Rank 5'])


rule67 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['ze']) &
                               (lbp_v['ze']  & dog_v['ze'])),
                  consequent=output['Rank 3'])

rule68 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['ze']) &
                               (lbp_v['ze']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule69 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['ze']) &
                               (lbp_v['pm']  & dog_v['nm'])),
                  consequent=output['Rank 5'])

rule70 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['ze']) &
                               (lbp_v['pm']  & dog_v['ze'])),
                  consequent=output['Rank 3'])

rule71 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['ze']) &
                               (lbp_v['pm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])


rule72 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['pm']) &
                               (lbp_v['nm']  & dog_v['nm'])),
                  consequent=output['Rank 5'])

rule73 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['pm']) &
                               (lbp_v['nm']  & dog_v['ze'])),
                  consequent=output['Rank 3'])


rule74 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['pm']) &
                               (lbp_v['nm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule75 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['pm']) &
                               (lbp_v['ze']  & dog_v['nm'])),
                  consequent=output['Rank 3'])

rule76 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['pm']) &
                               (lbp_v['ze']  & dog_v['ze'])),
                  consequent=output['Rank 3'])

rule77 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['pm']) &
                               (lbp_v['ze']  & dog_v['pm'])),
                  consequent=output['Rank 3'])

rule78 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['pm']) &
                               (lbp_v['pm']  & dog_v['nm'])),
                  consequent=output['Rank 3'])


rule79 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['pm']) &
                               (lbp_v['pm']  & dog_v['ze'])),
                  consequent=output['Rank 3'])

rule80 = ctrl.Rule(antecedent = ((hog_v['pm'] & orb_v['pm']) &
                               (lbp_v['pm']  & dog_v['pm'])),
                  consequent=output['Rank 3'])






system = ctrl.ControlSystem(
    rules=[rule0, rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14,
           rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27,
           rule28, rule29, rule30, rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40,
           rule41, rule42, rule43, rule44, rule45, rule46, rule47, rule48, rule49, rule50, rule51, rule52, rule53,
           rule54, rule55, rule56, rule57, rule58, rule59, rule60, rule61, rule62, rule63, rule64, rule65, rule66,
           rule67, rule68, rule69, rule70, rule71, rule72, rule73, rule74, rule75, rule76, rule77, rule78, rule79,
           rule80])


# system.view()
# Later we intend to run this system with a 21*21 set of inputs, so we allow
# that many plus one unique runs before results are flushed.
# Subsequent runs would return in 1/8 the time!
# sim = ctrl.ControlSystemSimulation(system, flush_after_run= 21 * 21 + 1)
sim = ctrl.ControlSystemSimulation(system)

def control_output(hog, orb, lbp, dog):
    sim.input['HOG'] = hog
    sim.input['ORB'] = orb
    sim.input['LBP'] = lbp
    sim.input['DOG'] = dog

    sim.compute()
    return sim.output['output']
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
# sim.input['HOG'] = 0.55
# sim.input['ORB'] = 0.27
# sim.input['LBP'] = 0.8
# sim.input['DOG'] = 0.9




# print(control_output(0.38, 0.7, 0.6, 0.87))
# Crunch the numbers
# sim.compute()

# print(sim.output['output'])
# output.view(sim=sim)
# plt.savefig('output_image_ex.png')
# plt.show()
# print(rule21.label)

# We can simulate at higher resolution with full accuracy
upsampled = np.linspace(-2, 2, 21)
x, y = np.meshgrid(upsampled, upsampled)
z = np.zeros_like(x)

# Loop through the system 21*21 times to collect the control surface
for i in range(21):
    for j in range(21):
        sim.input['HOG'] = x[i, j]
        sim.input['ORB'] = y[i, j]
        sim.input['LBP'] = x[i, j]
        sim.input['DOG'] = y[i, j]
        sim.compute()
        z[i, j] = sim.output['output']
# Plot the result in pretty 3D with alpha blending
# import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', linewidth = 0.4, antialiased = True)
cset = ax.contourf(x, y, z, zdir='z', offset=-2.5, cmap='viridis', alpha=0.5)
cset = ax.contourf(x, y, z, zdir='x', offset=3, cmap='viridis', alpha=0.5)
cset = ax.contourf(x, y, z, zdir='y', offset=3, cmap='viridis', alpha=0.5)

# plt.savefig("control_surface_3.png")
# ax.view_init(30, 200)
ax.view_init(40, 200)
plt.show()
