#import seaborn as sb
import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sb

data = np.array([[119, 174, 171, 306],
                [224, 170, 227, 211],
                [181, 206, 1089, 2709]])

data1 = np.array([[114.6180, 138.8943, 156.1132, 419.9824],
                  [140.6101, 145.7766, 162.764, 196.9213],
                  [165.7539, 167.3621, 235.5438, 719.0826]])

data2 = np.array([[186.0545, 144.6505, 318.8873, 202.2994],
                  [121.8016, 2396.5137, 2706.6440, 400.5080],
                  [134.9009, 185.3210, 5018.1919, 203.1913]])

plt.subplots(figsize=(20, 20))
#sb.palplot(sb.color_palette("ch:2.5,-.2,dark=.3"))
heatmap1 = sb.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Average Travel Time'})
#heatmap2 = sb.heatmap(data1, )
#heatmap3 = sb.heatmap(data2, )
plt.xticks(np.arange(4), [3, 4, 6, 8])
plt.yticks(np.arange(3), [0.4, 0.2, 0.05])
sb.set(font_scale=0.1)
plt.xlabel('Number of Traffic Intersections')
plt.ylabel('Traffic Congestion')
plt.savefig('Brute_HM', dpi=300)

