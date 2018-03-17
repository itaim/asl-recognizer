from asl_data import AslDb
import numpy as np

asl = AslDb() # initializes the database

asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']

# training = asl.build_training(features_ground)
# print("Training words: {}".format(training.words))
# training.get_word_Xlengths('CHOCOLATE')

df_means = asl.df.groupby('speaker').mean()
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
df_std = asl.df.groupby('speaker').std()

asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['speaker'].map(df_means['right-x']))/asl.df['speaker'].map(df_std['right-x'])
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['speaker'].map(df_means['right-y']))/asl.df['speaker'].map(df_std['right-y'])
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['speaker'].map(df_means['left-x']))/asl.df['speaker'].map(df_std['left-x'])
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['speaker'].map(df_means['left-y']))/asl.df['speaker'].map(df_std['left-y'])

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']

asl.df['polar-rr'] = ((asl.df['grnd-rx'])**2+(asl.df['grnd-ry'])**2)**0.5
asl.df['polar-rtheta'] = np.arctan2([asl.df['grnd-rx']],[asl.df['grnd-ry']])[0]
asl.df['polar-lr'] = ((asl.df['grnd-lx'])**2+(asl.df['grnd-ly'])**2)**0.5
asl.df['polar-ltheta'] = np.arctan2([asl.df['grnd-lx']],[asl.df['grnd-ly']])[0]

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']

asl.df['delta-rx'] = asl.df['grnd-rx'].diff(periods=1).fillna(0)
asl.df['delta-ry'] = asl.df['grnd-ry'].diff(periods=1).fillna(0)
asl.df['delta-lx'] = asl.df['grnd-lx'].diff(periods=1).fillna(0)
asl.df['delta-ly'] = asl.df['grnd-ly'].diff(periods=1).fillna(0)

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']

# asl.df['delta-polar-rr'] = asl.df['polar-rr'].diff(periods=1).fillna(0)
# asl.df['delta-polar-rtheta'] = asl.df['polar-rtheta'].diff(periods=1).fillna(0)
# asl.df['delta-polar-lr'] = asl.df['polar-lr'].diff(periods=1).fillna(0)
# asl.df['delta-polar-ltheta'] = asl.df['polar-ltheta'].diff(periods=1).fillna(0)
#
# features_polar_delta = ['delta-polar-rr','delta-polar-rtheta','delta-polar-lr','delta-polar-ltheta']

# asl.df['delta-polar-rr'] = asl.df['polar-rr'].diff(periods=1).fillna(0)
# asl.df['delta-polar-rtheta'] = asl.df['polar-rtheta'].diff(periods=1).fillna(0)
# asl.df['delta-polar-lr'] = asl.df['polar-lr'].diff(periods=1).fillna(0)
# asl.df['delta-polar-ltheta'] = asl.df['polar-ltheta'].diff(periods=1).fillna(0)

asl.df['gnorm-rx'] = (asl.df['grnd-rx'] - asl.df['speaker'].map(df_means['grnd-rx']))/asl.df['speaker'].map(df_std['grnd-rx'])
asl.df['gnorm-ry'] = (asl.df['grnd-ry'] - asl.df['speaker'].map(df_means['grnd-ry']))/asl.df['speaker'].map(df_std['grnd-ry'])
asl.df['gnorm-lx'] = (asl.df['grnd-lx'] - asl.df['speaker'].map(df_means['grnd-lx']))/asl.df['speaker'].map(df_std['grnd-lx'])
asl.df['gnorm-ly'] = (asl.df['grnd-ly'] - asl.df['speaker'].map(df_means['grnd-ly']))/asl.df['speaker'].map(df_std['grnd-ly'])

# TODO define a list named 'features_custom' for building the training set
features_custom = ['gnorm-rx','gnorm-ry','gnorm-lx','gnorm-ly']
# motion_delta = 2
# asl.df['hand-velocity-r'] = (asl.df['delta-rx'] ** 2 + asl.df['delta-ry'] ** 2) ** 0.5
# asl.df['hand-velocity-l'] = (asl.df['delta-lx'] ** 2 + asl.df['delta-ly'] ** 2) ** 0.5
# asl.df['hand-trajectory-r'] =  asl.df['hand-velocity-r'].cov(asl.df['hand-velocity-r'],min_periods=2).fillna(0)
# asl.df['hand-trajectory-l'] =  asl.df['hand-velocity-l'].cov(asl.df['hand-velocity-l'],min_periods=2).fillna(0)
# features_motion = ['hand-velocity-r', 'hand-velocity-l','hand-trajectory-r','hand-trajectory-l']


asl.df['dgnorm-rx'] = asl.df['gnorm-rx'].diff(periods=1).fillna(0)
asl.df['dgnorm-ry'] = asl.df['gnorm-ry'].diff(periods=1).fillna(0)
asl.df['dgnorm-lx'] = asl.df['gnorm-lx'].diff(periods=1).fillna(0)
asl.df['dgnorm-ly'] = asl.df['gnorm-ly'].diff(periods=1).fillna(0)

features_dgnorm = ['dgnorm-rx','dgnorm-ry','dgnorm-lx','dgnorm-ly']