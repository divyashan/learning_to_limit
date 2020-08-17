import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from utils.dataset_helpers import get_dataset
from scipy.sparse import csr_matrix
import pdb

class Dataset(object):
    def __init__(self, name, config):
        self.X = get_dataset(name)
        self.n_us = self.X.shape[0]
        self.n_fs = self.X.shape[1]
        
        self.checks = config['checks']
        init_pct = config['init_pct']
        test_pct = config['test_pct']
        init_mode = config['init_mode']
        
        user_idxs = range(self.n_us)
        feat_idxs = range(self.n_fs)
        all_idxs = list(product(user_idxs, feat_idxs))
        self.n_idxs = len(all_idxs)
        
        train_pct = 1-test_pct
        pool_idxs, test_idxs = train_test_split(all_idxs, test_size=test_pct)
        val_idxs, test_idxs = train_test_split(test_idxs, test_size=.5)
        init_idxs, pool_idxs = self.split(pool_idxs, config)
        
        self.init_idxs = set(init_idxs)
        self.pool_idxs = set(pool_idxs)
        self.val_idxs = set(val_idxs)
        self.test_idxs = set(test_idxs)
        self.acqu_idxs = set([])
        
        self.check_idxs()

    def split(self, idxs, config):
        pct = config['init_pct']/(1-config['test_pct'])
        mode = config['init_mode']
        avail_feats = range(self.n_fs)
        avail_users = range(self.n_us)
        if mode == 'features':
            feat_pct = config['feat_pct']
            n_feats = int(feat_pct*self.n_fs)
            avail_feats = np.random.choice(avail_feats, n_feats, replace=False)
        elif mode == 'users':
            user_pct = config['user_pct']
            n_users = int(user_pct*self.n_us)
            avail_users = np.random.choice(avail_users, n_users, replace=False)
        
        candidates = []
        n_choices = int(len(idxs)*pct)
        for idx in idxs:
            if idx[0] in avail_users and idx[1] in avail_feats:
                candidates.append(idx)
        cand_idxs = np.random.choice(range(len(candidates)), 
                                     n_choices, replace=False)
        init_idxs = [candidates[i] for i in cand_idxs]
        pool_idxs = list(set(idxs).difference(set(init_idxs)))
        return init_idxs, pool_idxs

    def add(self, idxs):
        self.acqu_idxs = self.acqu_idxs.union(set(idxs))
        self.pool_idxs = self.pool_idxs.difference(set(idxs)) 
        
        self.check_not_test(idxs)
        self.check_idxs()

    def available_idxs(self):
        return self.init_idxs.union(self.acqu_idxs)

    def observed_mask(self):
        mask = np.zeros(self.X.shape)
        observed_idxs = self.available_idxs() 
        for idx in observed_idxs:
            mask[idx[0],idx[1]] = 1
        return mask
        
    def n_available_idxs(self):
        return len(self.available_idxs())

    def observable_idxs(self):
        return self.available_idxs().union(self.pool_idxs)

    def n_observable_idxs(self):
        return self.n_available_idxs() + len(self.pool_idxs)

    def pct_available_idxs(self):
        n_observable = self.n_observable_idxs()
        n_available = self.n_available_idxs()
        return n_available/n_observable
    
    def cluster(self, k):
        # cluster based on initialized idxs
        # returns k groups of user ids
        row_inds = np.array(self.init_idxs)[:,0]
        col_inds = np.array(self.init_idxs)[:,1]
        data = [self.X[row_inds[i]][col_inds[i]] for i in range(len(row_inds))]
        csr_mat = csr_matrix((data, (row_inds, col_inds)), shape=(self.n_us, self.n_fs))
        kmeans = KMeans(n_clusters=k).fit(csr_mat)
        pred_clusters = kmeans.predict(csr_mat)
        return [np.where(pred_clusters == i)[0] for i in range(k)]

    def check_idxs(self):
        if self.checks:
            n_idxs = len(self.init_idxs) + len(self.pool_idxs) + \
                     len(self.test_idxs) + len(self.acqu_idxs) + \
                     len(self.val_idxs)
            assert n_idxs == self.n_idxs

    def check_not_test(self, idx):
        if self.checks:
            assert idx not in self.test_idxs

class MovieLensDataset(Dataset):
    def __init__(self, dataset_name, config): 
        self.X = self.get_mat(dataset_name) 
        self.checks = config['checks']
        self.n_us = self.X.shape[0]
        self.n_fs = self.X.shape[1]
        
        init_pct = config['init_pct']
        split_num = config['split_num']
        train_idxs, test_idxs = self.get_idxs(dataset_name, split_num)
        self.n_idxs = len(train_idxs) + len(test_idxs)
        n_init = int(init_pct*self.n_idxs) 

        val_idxs, test_idxs = train_test_split(test_idxs, test_size=.5)
        init_idxs, pool_idxs = self.split(train_idxs, config) 
        
        self.init_idxs = set(init_idxs)
        self.pool_idxs = set(pool_idxs)
        self.val_idxs = set(val_idxs)
        self.test_idxs = set(test_idxs)
        self.acqu_idxs = set([])
    
    def split(self, idxs, config):
        init_pct = config['init_pct']
        if config['init_mode'] == 'uniform':
            init_idxs, pool_idxs = train_test_split(idxs, 
                                                    train_size=init_pct/.8)
        elif config['init_mode'] != 'uniform':
            k_idxs, u_idxs = [], []
            if config['init_mode'] == 'user_subset':
                user_pct = config['user_pct']
                users = list(set(np.array(idxs)[:,0]))
                k_users, u_users = train_test_split(users, train_size=user_pct)
                print("Split users") 
                # organize idxs as map of 
                uid_mids_map = {idx[0]: [] for idx in idxs}
                for idx in idxs:
                    uid_mids_map[idx[0]].append(idx[1])
                print("Mapped uid to mids") 
                k_idxs = []
                for uid in k_users:
                    mids = uid_mids_map[uid]
                    k_idxs.extend([(uid, mid) for mid in mids])
                print("Built known idxs")

                u_idxs = []
                for uid in u_users:
                    mids = uid_mids_map[uid]
                    u_idxs.extend([(uid, mid) for mid in mids])
                print("Built unknown idxs")
                
            elif config['init_mode'] == 'item_subset':
                item_pct = config['item_pct']
                items = list(set(np.array(idxs)[:,1]))
                k_items, u_items = train_test_split(items, train_size=item_pct)
                mid_uids_map = {idx[1]: [] for idx in idxs}
                
                for idx in idxs:
                    mid_uids_map[idx[1]].append(idx[0])

                k_idxs = []
                for mid in k_items:
                    uids = mid_uids_map[mid]    
                    k_idxs.extend([(uid, mid) for uid in uids])

                u_idxs = []
                for mid in u_items:
                    uids = mid_uids_map[mid]
                    u_idxs.extend([(uid, mid) for uid in uids])
                
            pct_known = len(k_idxs)/(len(idxs))
            calc_pct = (init_pct/.8)/pct_known
            init_idxs, pool_idxs = train_test_split(k_idxs, train_size=calc_pct)
            pool_idxs = np.concatenate([pool_idxs, u_idxs])
        init_idxs = list(map(tuple, init_idxs))
        pool_idxs = list(map(tuple, pool_idxs))
        return init_idxs, pool_idxs

    def get_funksvd_df(self, idxs):
        # Return train df for FunkSVD
        rows = []
        for idx in idxs:
            rows.append({'u_id': idx[0], 'i_id': idx[1], 
                         'rating': self.X[idx[0], idx[1]]})
        return pd.DataFrame(rows)

    def get_mat(self, dataset_name):
        ratings = pd.read_csv('datasets/' + dataset_name + '/u.data') 
        ratings_matrix = ratings.pivot_table(index=['uid'],columns=['mid'],values='rating').reset_index(drop=True)
        ratings_matrix.fillna(0, inplace = True)
        data_matrix = np.array(ratings_matrix)
        return data_matrix
    
    def get_idxs(self, dataset_name, split_num):
        tr_ratings = pd.read_csv('datasets/' + dataset_name + '/u' + str(split_num) + '.base')
        test_ratings = pd.read_csv('datasets/' + dataset_name + '/u' + str(split_num) + '.test')
        train_idxs = tr_ratings[['uid', 'mid']].values 
        test_idxs = test_ratings[['uid', 'mid']].values 
        #train_idxs = train_idxs - 1
        #test_idxs = test_idxs - 1
        train_idxs = list(map(tuple, train_idxs))
        test_idxs = list(map(tuple, test_idxs))
        return train_idxs, test_idxs

class UsrGrpDataset(Dataset):

    def __init__(self, dataset, user_idxs):
        self.user_idxs = user_idxs
        
        self.X = dataset.X[self.user_idxs,:]
        self.n_us = self.X.shape[0]
        self.n_fs = self.X.shape[1]
        self.local_map = {user_idxs[i]:i for i in range(self.n_us)}

        self.n_idxs = self.n_us*self.n_fs
        self.checks = dataset.checks

        self.init_idxs = self.to_local(dataset.init_idxs) 
        self.pool_idxs = self.to_local(dataset.pool_idxs)
        self.val_idxs = self.to_local(dataset.val_idxs)
        self.test_idxs = self.to_local(dataset.test_idxs)
        self.acqu_idxs = [] 

    def to_local(self, global_idxs):
        return [(self.local_map[i[0]], i[1]) for i in global_idxs if (i[0] in self.user_idxs)]

    def to_global(self, local_idxs):
        return [(self.user_idxs[i[0]], i[1]) for i in local_idxs]
