from eval_feat import eval_feat
import scipy.io

features = scipy.io.loadmat('/home/zz/PGCN/features/market1501.mat')  # the path of features, '.mat' file
features['q_label'] = features['q_label'][0]
features['g_label'] = features['g_label'][0]
features['q_cam'] = features['q_cam'][0]
features['g_cam'] = features['g_cam'][0]
results = eval_feat(features)
