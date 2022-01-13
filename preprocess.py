import time
import csv
import pickle
import operator
import datetime
import os

class Preprocess():
    def __init__(self,dataset = 'datasets\sample_train-item-views.csv'):
        self.dataset = dataset
    
    def start(self):
        item_dict = {}
        print("-- Starting @ %ss" % datetime.datetime.now())
        with open(self.dataset, "r") as f:
            reader = csv.DictReader(f, delimiter=';')
            sess_clicks = {}
            sess_date = {}
            ctr = 0
            curid = -1
            curdate = None
            for data in reader:
                sessid = data['session_id']
                if curdate and not curid == sessid:
                    date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
                    sess_date[curid] = date
                curid = sessid
                item = data['item_id'], int(data['timeframe'])
                curdate = data['eventdate']
                if sessid in sess_clicks:
                    sess_clicks[sessid] += [item]
                else:
                    sess_clicks[sessid] = [item]
                ctr += 1
            date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
            for i in list(sess_clicks):
                sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))
                sess_clicks[i] = [c[0] for c in sorted_clicks]
            sess_date[curid] = date
        print("-- Reading data @ %ss" % datetime.datetime.now())
        
        #filter out length 1 sessions
        sess_clicks, sess_date = self.filter_length_sessions(sess_clicks, sess_date)
        
        # count number of times each item appears
        iid_counts = self.count_number_of_times_each_item_appears(sess_clicks)

        #take top 5 click
        sess_clicks, sess_date = self.filter_top_5(sess_clicks,sess_date,iid_counts)

        #max date for split
        maxdate, dates = self.get_max_date(sess_date)

        # 7 days for test
        splitdate = maxdate - 86400 * 7

        #split data train test
        train_sess, test_sess = self.spit_data(dates,splitdate)      # Yoochoose: ('Split date', 1411930799.0)
        train_sess, test_sess = self.sort_sessions_by_date(train_sess, test_sess)
        
        train_ids, train_dates, train_seqs = self.obtian_train(train_sess,sess_clicks,item_dict)
        test_ids, test_dates, test_seqs = self.obtian_test(test_sess,sess_clicks,item_dict)

        tr_seqs, tr_dates, train_labs, tr_ids = self.process_seqs(train_seqs, train_dates)
        te_seqs, te_dates, test_labs, te_ids = self.process_seqs(test_seqs, test_dates)

        train = (tr_seqs, train_labs)
        test = (te_seqs, test_labs)

        all = 0

        for seq in train_seqs:
            all += len(seq)
        for seq in test_seqs:
            all += len(seq)
        print('avg length: ', all/(len(train_seqs) + len(test_seqs) * 1.0))
        
        #write to file train and test
        if not os.path.exists('sample'):
            os.makedirs('sample')
        pickle.dump(train, open('sample/train.txt', 'wb'))
        pickle.dump(test, open('sample/test.txt', 'wb'))
        pickle.dump(train_seqs, open('sample/all_train_seq.txt', 'wb'))

        print('pre-processing is done.')

    def filter_length_sessions(self,sess_clicks, sess_date):
        for s in list(sess_clicks):
            if len(sess_clicks[s]) == 1:
                del sess_clicks[s]
                del sess_date[s]
        return sess_clicks, sess_date

    def count_number_of_times_each_item_appears(self,sess_clicks):
        iid_counts = {}
        for s in sess_clicks:
            seq = sess_clicks[s]
            for iid in seq:
                if iid in iid_counts:
                    iid_counts[iid] += 1
                else:
                    iid_counts[iid] = 1
        return iid_counts

    def filter_top_5(self,sess_clicks,sess_date,iid_counts):
        for s in list(sess_clicks):
            curseq = sess_clicks[s]
            filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))
            if len(filseq) < 2:
                del sess_clicks[s]
                del sess_date[s]
            else:
                sess_clicks[s] = filseq
        return sess_clicks, sess_date   

    def get_max_date(self,sess_date):
        dates = list(sess_date.items())
        maxdate = dates[0][1]
        for _, date in dates:
            if maxdate < date:
                maxdate = date
        return maxdate, dates

    def spit_data(self,dates,splitdate):
        print('Splitting date', splitdate)
        train_sess = filter(lambda x: x[1] < splitdate, dates)
        tesr_sess = filter(lambda x: x[1] > splitdate, dates)
        return train_sess, tesr_sess

    def sort_sessions_by_date(self,train_sess, test_sess):
        train_sess = sorted(train_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
        test_sess = sorted(test_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
        print(len(train_sess))    # 186670    # 7966257
        print(len(test_sess))    # 15979     # 15324
        print(train_sess[:3])
        print(test_sess[:3])
        return train_sess, test_sess
    
    def obtian_train(self,train_sess,sess_clicks,item_dict):
        train_ids = []
        train_seqs = []
        train_dates = []
        item_ctr = 1
        for s, date in train_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
                else:
                    outseq += [item_ctr]
                    item_dict[i] = item_ctr
                    item_ctr += 1
            if len(outseq) < 2:  # Doesn't occur
                continue
            train_ids += [s]
            train_dates += [date]
            train_seqs += [outseq]
        print(item_ctr)     # 43098, 37484
        return train_ids, train_dates, train_seqs

    def obtian_test(self,test_sess,sess_clicks,item_dict):
        test_ids = []
        test_seqs = []
        test_dates = []
        for s, date in test_sess:
            seq = sess_clicks[s]
            outseq = []
            for i in seq:
                if i in item_dict:
                    outseq += [item_dict[i]]
            if len(outseq) < 2:
                continue
            test_ids += [s]
            test_dates += [date]
            test_seqs += [outseq]
        return test_ids, test_dates, test_seqs
    
    def process_seqs(self,iseqs, idates):
        out_seqs = []
        out_dates = []
        labs = []
        ids = []
        for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
            for i in range(1, len(seq)):
                tar = seq[-i]
                labs += [tar]
                out_seqs += [seq[:-i]]
                out_dates += [date]
                ids += [id]
        return out_seqs, out_dates, labs, ids   