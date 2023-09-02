#!/usr/bin/env python3
"""
Provides aerodynamic data of the Bebop2 identified in the wind tunnel at TUDelft

It contains the aerodynamic parameters for the Matlab model computations
"""

import numpy as np

__author__ = "Jose Ignacio de Alvear Cardenas (GitHub: @joigalcar3)"
__copyright__ = "Copyright 2022, Jose Ignacio de Alvear Cardenas"
__credits__ = ["Jose Ignacio de Alvear Cardenas"]
__license__ = "MIT"
__version__ = "1.0.1 (04/04/2022)"
__maintainer__ = "Jose Ignacio de Alvear Cardenas"
__email__ = "jialvear@hotmail.com"
__status__ = "Stable"

# Identified polynomial coefficients for the computation of the thrust coefficient
k_Ct0 = np.array([[0.0152457017219075, -3.19835880466424e-05, -0.0474659629880834, -5.48089604291955e-08,
                  -0.000164550969624146, 0.650877249185920, 1.10477778832442e-08, -9.76752919452344e-06,
                  0.00859691522825337, -2.20418122442645, 3.27434126987218e-11, -5.69117054658112e-08,
                  2.32561854294217e-05, -0.0116550184566165, 3.04959484433102, -6.00795185558617e-13,
                  1.81690349314076e-10, -4.63671043348055e-08, -1.52454780569063e-05, 0.00607313609112646,
                  -1.51563942225535]])

# Identified polynomial coefficients for the computation of the thrust coefficient correction
k_model_2 = np.array([[0.00274750362242118, 0.0587325418628517, 0.0291979006795257, 0.155176381977433,
                      -0.848919654295447, -2.85285652127970, -16.6872606424138, -25.3582092758054,
                      -8.21139486023900, 0.000662666074942486, -0.0137515544490184, 0.0258468383560923,
                      0.129354278831284, 0.739022953068088, -0.751589665006720, 2.30225029828906, 3.19387485842342,
                      -2.01400124775404, -0.0327090226346993, 0.00550048663001955, 0.704747484311295,
                      0.384042859177342, 0.409107040799852, -2.91693142809590, -5.72731924259749, -3.84424311448819,
                      0.957068915766478, 0.00798767042006989, -0.0658174319226979, -0.515362725845307,
                      0.154017181898898, 1.07229345471127, 5.60834749404815, 3.12341580631406, 13.2751387862931,
                      3.38340384818304, -0.00871325200163225, 0.0139319909808224, 0.135829051260073,
                      0.0724018634704221, 0.462231305873754, 1.07728548843851, -2.92439099099261, 2.07387265629944,
                      -1.76236683822441, 0.00277901355913424, 5.93712461960435e-05, -0.0737682036851551,
                      0.408392701436168, 0.181780336855863, -0.0914796558508702, -5.33048488631146,
                      -11.6294693255163, -4.72950404100762, -0.00594871416216384, -0.0162850806730608,
                      0.173368295316786, 0.186292675296392, 0.225644067201894, -0.688845939593434, -6.49432628543192,
                      -7.80900137821226, 0.415239218701371, -0.00544216811616573, 0.00518487316578840,
                      0.0476580090813803, -0.200801241660794, -0.476117215479456, -0.407991135460875,
                      -1.81735072025647, 1.50472930028764, 4.35662490484023, -0.00159368739623987,
                      0.000467723919419556, 0.0129022985413385, -0.142747208717601, -0.286423056758624,
                      -0.233246678589007, 5.27930446169201, 6.06363387971617, 3.14128857337644, 0.00453268191002699,
                      -0.00474962613583822, -0.180460224377998, -0.0116017180130748, 0.0192198318847662,
                      1.17708508701190, 0.0640467785184096, 3.10723451211166, 0.482465692101886]]).T

# Identified polynomial coefficients for the computation of the torque coefficient
k_Cq0 = np.array([[-0.000166978387654207, -9.26661647846620e-07, -0.000161106517356852, 1.49219451037256e-09,
                   -2.80468068962665e-06, 0.000591396065463947, 4.46363200546300e-10, 8.90349145739088e-08,
                   -1.53880349952214e-05, -0.00773976740405967, -3.70391296926118e-13, 3.92836511888492e-10,
                   -1.33297247718639e-08, -0.000133549679393062, 0.0164947421115812, -4.17586454158575e-14,
                   -3.24864794974322e-12, 1.14205811263298e-09, -6.42233810561959e-08, 0.000149532607712236,
                   -0.0106110955476936]])

# Identified polynomial coefficients for the computation of the torque coefficient correction
k_model_11 = np.array([[0.00260329204354066, 0.00129328992586148, -0.0199809965492002, -0.0868022523710462,
                        -0.0889469386700429, 0.128032771798353, 0.146886709138850, 0.524080931866815,
                        0.725843471299357, 0.00242937984350116, -0.00310550867822261, -0.0595768021706452,
                        0.0523222113704624, 0.0799385477524136, 0.0802433226135493, 0.387865290852874,
                        0.137337555435232, -0.310321282567866, 0.00158155254379188, 0.00319704874615568,
                        -0.0554013498676434, -0.133883081875846, -0.154861909695999, 0.246917675390354,
                        0.0807388330437734, 1.21408540541708, 0.991981880292023, -9.00820660019962e-05,
                        0.000417739287775632, -0.00176625889617756, -0.0352354581017755, 0.00347023118130632,
                        -0.0818779712415576, -0.0939352353976481, 0.367104057232038, -0.239846676494934,
                        0.000548702590051651, 0.000217200486637933, -0.00564899836836745, -0.000264397140190192,
                        0.00896129402789547, -0.0942019724552947, 0.0408551476683280, 0.607021266741172,
                        0.0144874105803823, 0.000847230370183477, -0.000948252583147180, -0.00265605533100469,
                        0.0598956109168678, 0.0807953120897514, -0.0545778293654141, -0.235368707057857,
                        -0.948022031763549, -0.608815444932934, -0.000182959785330574, 0.00167139842657429,
                        0.00833552390363391, 0.0167067780351973, -0.00216159653990414, -0.0668352653475071,
                        0.0332682896037231, -0.220002714254035, -0.100740744918869, -0.000459348737065979,
                        -0.00323901937377689, 0.0150989940672047, -0.00337931488830346, 0.0705271437626767,
                        -0.0355357034004435, -0.0945407727921580, -0.114237238851565, -0.348109605269992,
                        -0.000423056934179010, -1.57962640331396e-05, 0.00625198744041810, -0.0204741957877981,
                        -0.00655890452523322, -0.0335286749006157, 0.0745650825531103, 0.0289036906676707,
                        -0.0296758936977819, -0.000231091410645385, -0.000748657303930713, 0.00208598921233990,
                        -0.0132023573075178, 0.0116429676033409, 0.0151697188209396, -0.0404565223580964,
                        0.178329482629743, 0.0297489549373982]]).T