import time
import os
import datetime
import pymssql
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.simplefilter("ignore")



def get_sql(sql_path, file_nm):
    wk_dir = os.path.dirname(os.path.realpath(__file__))
    sql_path = os.path.join(wk_dir , sql_path)
    tmp_file = os.path.join(sql_path, file_nm)

    with open(tmp_file, encoding='utf-8') as f:
        sql = f.read()
    return sql

def f(col1, col2, col3 ):
    if col3 == 1:
        return col1
    else :
        return col2

class VolStrategy:

    def __init__(self, name, std_date, pf_t_ratio):
        
        self.name = name
        self.pf_t_ratio = pf_t_ratio
        self.testtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        self.sql_path = './sql'
        self.idx_p = 'kr_index_price1.sql'
        self.stck_p = 'kr_stock_price1.sql'
        self.cur_idx_p = 'kr_index_cur_price.sql'
        self.cur_stck_p = 'kr_stock_cur_price.sql'

        self.code_list = [['KR7069500007','A069500','KODEX 200']
             ,['KR7102110004','A102110','TIGER 200']
             ,['KR7148020001','A148020','KBSTAR 200']
             ,['KR7130730005','A130730','KOSEF 단기자금']
             ,['KR7196230007','A196230','KBSTAR 단기통안채']
             ,['KR7272580002','A272580','TIGER 단기채권액티브']
             ]
        
        self.s_col = ['KODEX 200','TIGER 200','KBSTAR 200']
        self.b_col = ['KOSEF 단기자금','KBSTAR 단기통안채','TIGER 단기채권액티브']


        self.idx_sql = get_sql(self.sql_path, self.idx_p)
        self.stck_sql = get_sql(self.sql_path, self.stck_p)

        self.idx_sql = self.idx_sql.format(from_date=std_date)
        self.stck_sql = self.stck_sql.format(from_date=std_date)

        self.cur_idx_sql = get_sql(self.sql_path, self.cur_idx_p)
        self.cur_stck_sql = get_sql(self.sql_path, self.cur_stck_p)

        self.conn = pymssql.connect(host='10.93.20.65', user='quant', password='mirae', database='MARKET', charset='utf8')
        self.idx_df = pd.read_sql(self.idx_sql, con=self.conn)
        self.stck_df = pd.read_sql(self.stck_sql, con=self.conn)
        self.idx_df.NAME = self.idx_df.NAME.map(lambda x : x.encode('ISO-8859-1').decode('euc-kr'))
        self.stck_df.NAME = self.stck_df.NAME.map(lambda x : x.encode('ISO-8859-1').decode('euc-kr'))
        
        self.idx_df.DATE = pd.to_datetime(self.idx_df.DATE)
        self.idx_df = self.idx_df.set_index('DATE')
        
        self.stck_df.DATE = pd.to_datetime(self.stck_df.DATE)
        self.stck_df = self.stck_df.set_index('DATE')
    
        
    def feat_datetime(self, cur_data) :
        cur_date = cur_data['RMS_DTM'][0][:8]
        cur_time = cur_data['RMS_DTM'][0][8:15]
        t_df = cur_data.iloc[:,1:].copy()
        t_df['DATE'] = pd.to_datetime(cur_date)
        t_df = t_df.set_index('DATE')
        return t_df

    def add_cur_data(self, name):


        self.cur_idx_df = pd.read_sql(self.cur_idx_sql, con=self.conn)
        self.cur_stck_df = pd.read_sql(self.cur_stck_sql, con=self.conn)

        self.cur_idx_df = self.feat_datetime( self.cur_idx_df )
        self.cur_stck_df = self.feat_datetime( self.cur_stck_df )
        
        self.cur_idx_df = pd.concat([self.idx_df,self.cur_idx_df])
        self.cur_stck_df = pd.concat([self.stck_df,self.cur_stck_df])

    def calc_mp(self, name):
        
        V = 0.15 # 변동성(0.15 고정)
        M = 5    # 만기(5년 고정)

        if name == "BACKTEST":
            s_df = self.stck_df.copy()
            i_df = self.idx_df.copy()
        else :
            s_df = self.cur_stck_df.copy()
            i_df = self.cur_idx_df.copy()
            

        df_list = []
        for std_code, code, name in self.code_list:
            cond1 = s_df.CODE == code
            df_list.append(s_df[cond1]['CLOSE_AP'])
        df = pd.concat(df_list, axis=1)
        df.columns = [name for std_code, code, name in self.code_list] 

        i_df = i_df[['CLOSE_P']].copy()
        i_df['k200_ret'] = i_df['CLOSE_P'].pct_change().fillna(0)
        i_df['roll_s_p'] = i_df['CLOSE_P'].rolling(250).max() * 1.2
        i_df['t_s_p'] = i_df['CLOSE_P'] * 1.2
        i_df['strike price'] = i_df[['roll_s_p','t_s_p']].max(axis=1) #K200 행사가

        t = ((np.log(i_df['CLOSE_P']/i_df['strike price']) + (( 0 + V**2) / 2 * 5)) / (V * np.sqrt(M)))
        i_df['trg_wei'] = 1 - stats.norm.cdf(t)
        i_df['bf_rebal_wei'] = 0 
        i_df['rebal_sig'] = (np.abs(i_df['trg_wei'] - i_df['bf_rebal_wei']) > 0.01).map(lambda x : 1 if x else 0)
        i_df['af_rebal_wei'] = i_df.apply(lambda x: f(x['trg_wei'], x['bf_rebal_wei'], x['rebal_sig']), axis=1)
        i_df['bf_rebal_wei'] = (i_df['af_rebal_wei'].shift(1) * (1+i_df['k200_ret'])).fillna(0)
        i_df['rebal_sig'] = (np.abs(i_df['trg_wei'] - i_df['bf_rebal_wei']) > 0.01).map(lambda x : 1 if x else 0)
        i_df['af_rebal_wei'] = i_df.apply(lambda x: f(x['trg_wei'], x['bf_rebal_wei'], x['rebal_sig']), axis=1)

        i_df['af_rebal_wei'] = i_df['af_rebal_wei'].mul(self.pf_t_ratio)
        
        r_df = df.pct_change().fillna(0).copy()
        s_eq_wei = i_df['af_rebal_wei'] / len(self.s_col)
        b_eq_wei = (1 - i_df['af_rebal_wei']) / len(self.b_col)

        s_d_ret = r_df[self.s_col].mul(s_eq_wei, axis=0).sum(axis=1)
        b_d_ret = r_df[self.b_col].mul(b_eq_wei, axis=0).sum(axis=1)
        tot_d_ret = s_d_ret +  b_d_ret

        for col_n in self.s_col:
            n_col = f"{col_n}_W"
            r_df[n_col] = s_eq_wei
        for col_n in self.b_col:
            n_col = f"{col_n}_W"
            r_df[n_col] = b_eq_wei
        
        r_df['af_rebal_wei'] = i_df['af_rebal_wei']
        r_df['ret'] = tot_d_ret
        r_df['cum_ret'] = (1+r_df['ret']).cumprod()
        r_df['BM_RET'] = (i_df['k200_ret'] + 1 ).cumprod()
        r_df.reset_index(inplace=True)

        return r_df
    
    

    def insert_mp(self, result_df, mp_type, sql):
        """
        TODO : 중복체크 쿼리 분리하면 좋을듯함.

        """
        r_cnt = 0 
        check_sql = """select count(*) from QUANTDEV..RATB_MP where 1=1 and DT = '{date}' and MP_TYPE = '{mp_type}' and CODE = '{code}' """
        cursor = self.conn.cursor()
        for i, rows in result_df.iterrows():
            for code_map in self.code_list:
                dt = rows['DATE'].strftime('%Y%m%d')
                pre_sql = check_sql.format(date = dt, 
                                           mp_type = mp_type, 
                                           code = code_map[1]
                                           )
                cursor.execute(pre_sql)
                cnt = cursor.fetchall()
                # 중복 키 체크
                if cnt[0][0] > 0 : 
                    pass
                else :
                    value = (dt, mp_type, code_map[1] ,code_map[2] , rows[code_map[2]] , rows[f"{code_map[2]}_W"])
                    cursor.execute(sql, value)
                    r_cnt += 1
                    # print(value)
        self.conn.commit()
        print(f'테이블 : QUANTDEV..RATB_MP  \n포트유형 : 변동성_{mp_type} \n적재 건수 : {r_cnt} 건 \n')

    

if __name__ == '__main__' :

    """
    1. mode에서 1가지를 옵션 선택
    2. trg_st, trg_et 모니터링 시간대 지정
    3. std_date 날짜 선택 
    4. 테스트 진행

    """
    # 1. mode선택
    # mode = 'LIVE'         # 일중 모니터링 모드
    # mode = 'CUR_TEST'   # 일중 테스트 모드
    mode = 'BACKTEST'   # 과거 테스트 모드

    # 2. 시간대 선택
    # trg_st = datetime.time(14,30,00)
    # trg_et = datetime.time(14,30,59)
    trg_st = datetime.time(11,32,00)
    trg_et = datetime.time(11,32,59)

    # 3. 조회날짜 선택
    # std_date = '20191229'
    std_date = (datetime.datetime.now() - datetime.timedelta(500))
    std_date = std_date.strftime('%Y%m%d')

    pf1 = VolStrategy(name='공격형', std_date=std_date, pf_t_ratio=1)   # 공격형 100%
    pf2 = VolStrategy(name='중립형', std_date=std_date, pf_t_ratio=0.8) # 중립형 80%
    pf3 = VolStrategy(name='안정형', std_date=std_date, pf_t_ratio=0.6) # 안정형 60%

    if mode  == 'LIVE' :
        print("========================")
        print("CUR DATA MONITORING MODE")
        print("========================")

        while True:
            try:
                now_t = datetime.datetime.now().time()
                print(now_t)
                pf1.add_cur_data(name=mode)
                pf2.add_cur_data(name=mode)
                pf3.add_cur_data(name=mode)
                result_df1 = pf1.calc_mp(name=mode)
                result_df2 = pf2.calc_mp(name=mode)
                result_df3 = pf3.calc_mp(name=mode)
                pf1_r = str(round(result_df1.tail(1)['af_rebal_wei'][0],4)* 100) + "%"
                pf2_r = str(round(result_df2.tail(1)['af_rebal_wei'][0],4)* 100) + "%"
                pf3_r = str(round(result_df3.tail(1)['af_rebal_wei'][0],4)* 100) + "%"
                print( pf1_r,' // ',pf2_r,' // ',pf3_r)
                
                if trg_st < now_t < trg_et:                
                    break
                time.sleep(5)
            except Exception as e:
                print(e)

    elif mode == 'CUR_TEST':
        print("========================")
        print("CUR DATA TEST MODE")
        print("========================")
        pf1.add_cur_data(name=mode)
        pf2.add_cur_data(name=mode)
        pf3.add_cur_data(name=mode)
        result_df1 = pf1.calc_mp(name=mode)
        result_df2 = pf2.calc_mp(name=mode)
        result_df3 = pf3.calc_mp(name=mode)

    else :
        print("========================")
        print("BACKTEST MODE")
        print("========================")
        result_df1 = pf1.calc_mp(name=mode)
        result_df2 = pf2.calc_mp(name=mode)
        result_df3 = pf3.calc_mp(name=mode)
    
    ################
    #### report ####
    ################
    sql = "INSERT INTO QUANTDEV..RATB_MP (DT, MP_TYPE, CODE, KN, RTN_D, WEI_D ) VALUES (%s, %s, %s, %s, %s, %s)"
    pf1.insert_mp(result_df1, '공격형', sql)
    pf2.insert_mp(result_df2, '중립형', sql)
    pf3.insert_mp(result_df3, '안정형', sql)
