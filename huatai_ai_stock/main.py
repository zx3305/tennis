import tushare
import pandas as pd
import os
from datetime import datetime, date, timedelta
import time

os.environ['TZ'] = 'Asia/Shanghai'

pro = tushare.pro_api("1bbf2b41a2180bc45022199c6ffdab5b85b07729562d189b22724b93")
'''
大类因子 具体因子 因子描述 因子方向
估值 EP 净利润（TTM）/总市值 1
估值 EPcut 扣除非经常性损益后净利润（TTM）/总市值 1
估值 BP 净资产/总市值 1
估值 SP 营业收入（TTM）/总市值 1
估值 NCFP 净现金流（TTM）/总市值 1
估值 OCFP 经营性现金流（TTM）/总市值 1
估值 DP 近 12 个月现金红利（按除息日计）/总市值 1
估值 G/PE 净利润（TTM）同比增长率/PE_TTM 1
成长 Sales_G_q 营业收入（最新财报，YTD）同比增长率 1
成长 Profit_G_q 净利润（最新财报，YTD）同比增长率 1
成长 OCF_G_q 经营性现金流（最新财报，YTD）同比增长率 1
成长 ROE_G_q ROE（最新财报，YTD）同比增长率 1
财务质量 ROE_q ROE（最新财报，YTD） 1
财务质量 ROE_ttm ROE（最新财报，TTM） 1
财务质量 ROA_q ROA（最新财报，YTD） 1
财务质量 ROA_ttm ROA（最新财报，TTM） 1
财务质量 grossprofitmargin_q 毛利率（最新财报，YTD） 1
财务质量 grossprofitmargin_ttm 毛利率（最新财报，TTM） 1
财务质量 profitmargin_q 扣除非经常性损益后净利润率（最新财报，YTD） 1
财务质量 profitmargin_ttm 扣除非经常性损益后净利润率（最新财报，TTM） 1
财务质量 assetturnover_q 资产周转率（最新财报，YTD） 1
财务质量 assetturnover_ttm 资产周转率（最新财报，TTM） 1
财务质量 operationcashflowratio_q 经营性现金流/净利润（最新财报，YTD） 1
财务质量 operationcashflowratio_ttm 经营性现金流/净利润（最新财报，TTM） 1
杠杆 financial_leverage 总资产/净资产 -1
杠杆 debtequityratio 非流动负债/净资产 -1
杠杆 cashratio 现金比率 1
杠杆 currentratio 流动比率 1
市值 ln_capital 总市值取对数 -1
动量反转 HAlpha 个股 60 个月收益与上证综指回归的截距项 -1
动量反转 return_Nm 个股最近 N 个月收益率，N=1，3，6，12 -1
动量反转 wgt_return_Nm 个股最近 N 个月内用每日换手率乘以每日收益率求算术平均
值，N=1，3，6，12
-1
动量反转 exp_wgt_return_Nm 个股最近 N 个月内用每日换手率乘以函数 exp(-x_i/N/4)再乘
以每日收益率求算术平均值，x_i 为该日距离截面日的交易日
的个数，N=1，3，6，12
-1
波动率 std_FF3factor_Nm 特质波动率——个股最近 N 个月内用日频收益率对 Fama 
French 三因子回归的残差的标准差，N=1，3，6，12
-1
波动率 std_Nm 个股最近 N 个月的日收益率序列标准差，N=1，3，6，12 -1
股价 ln_price 股价取对数 -1
beta beta 个股 60 个月收益与上证综指回归的 beta -1
换手率 turn_Nm 个股最近 N 个月内日均换手率（剔除停牌、涨跌停的交易日），
N=1，3，6，12
-1
换手率 bias_turn_Nm 个股最近 N个月内日均换手率除以最近 2年内日均换手率（剔
除停牌、涨跌停的交易日）再减去 1，N=1，3，6，12
-1
情绪 rating_average wind 评级的平均值 1
情绪 rating_change wind 评级（上调家数-下调家数）/总数 1
情绪 rating_targetprice wind 一致目标价/现价-1 1
股东 holder_avgpctchange 户均持股比例的同比增长率 1
技术 MACD 经典技术指标（释义可参考百度百科），长周期取 30 日，短
周期取 10 日，计算 DEA 均线的周期（中周期）取 15 日 -1
技术 DEA -1
技术 DIF -1
技术 RSI 经典技术指标，周期取 20 日 -1
技术 PSY 经典技术指标，周期取 20 日 -1
技术 BIAS 经典技术指标，周期取 20 日 -1
'''

def factor(trade_date, code):
	df = pro.daily_basic(ts_code=code, trade_date=trade_date.strftime('%Y%m%d'),\
		fields='pe_ttm,pb,ps_ttm,total_mv')
	data = pd.DataFrame(index=[0])
	data['EP'] = 1/df['pe_ttm'].astype("float")
	data['BP'] = 1/df['pb'].astype("float")
	data['SP'] = 1/df['ps_ttm'].astype("float")

	retDict = quarterQuota(trade_date, code)
	data['OCFP'] = retDict['net_cash_ttm']/df['total_mv']
	data['G/Profit_G_q'] = retDict['netprofit_yoy']
	data['Sales_G_q'] = retDict['or_yoy']
	data['OCF_G_q'] = retDict['ocf_yoy']
	data['ROE_G_q'] = retDict['roe_yoy']
	data['ROE_ttm'] = retDict['roe_yearly']
	data['ROA_ttm'] = retDict['roa2_yearly']
	data['grossprofitmargin_q'] = retDict['q_gsprofit_margin']
	data['grossprofitmargin_ttm'] = retDict['grossprofit_margin']
	data['profitmargin_q'] = retDict['profitmargin_q']
	print(data)


#获取一个季度内的指标数据
#net_cash_ttm 净现金流(TTM)
#n_cashflow_act 经营性现金流（TTM）
#netprofit_yoy 净利润（TTM）同比增长率
#or_yoy 营业收入同步增长
#ocf_yoy 经营活动产生的现金流量净额同比增长率(%)
#roe_yearly 年化净资产收益率
#roa2_yearly 年化总资产报酬率
#profitmargin_q 
def quarterQuota(cur_datetime, code):
	timeListTtm = getTTMYearAndQuarter(cur_datetime)
	timeListYtd = getTTMYearAndQuarter(cur_datetime, 'ytd')
	net_cash_ttm = 0
	netprofit_yoy = 0
	or_yoy = 0
	ocf_yoy = 0
	roe_yoy = 0
	roe_yearly = 0
	grossprofit_margin = 0
	q_gsprofit_margin = 0
	profitmargin_q = 0
	grossprofitmargin_ttm = 0
	retDict = {'net_cash_ttm':0}
	for tupTime in timeListTtm:
		df = pro.cashflow(ts_code=code, start_date=tupTime[0],end_date=tupTime[1],\
			fields='n_cashflow_act,n_cashflow_inv_act,n_cash_flows_fnc_act')
		# df2 = pro.fina_indicator(ts_code=code, start_date=tupTime[0],end_date=tupTime[1],\
		# 	fields='profit_dedt')
		if df.empty: 
			print("存在空值")
		else:
			retDict['net_cash_ttm'] += (df.loc[0, 'n_cashflow_act'] + df.loc[0, 'n_cashflow_inv_act'] + df.loc[0, 'n_cash_flows_fnc_act'])*tupTime[2]
	for tupTime in timeListYtd:
		df = pro.fina_indicator(ts_code=code, start_date=tupTime[0],end_date=tupTime[1],\
			fields='or_yoy,netprofit_yoy,ocf_yoy,roe_yoy,roe_yearly,roa2_yearly,\
			grossprofit_margin,q_gsprofit_margin,end_date,profit_dedt,q_dtprofit')
		retDict['or_yoy'] = df.loc[0, 'or_yoy']
		retDict['netprofit_yoy'] = df.loc[0, 'netprofit_yoy']
		retDict['ocf_yoy'] = df.loc[0, 'ocf_yoy']
		retDict['roe_yoy'] = df.loc[0, 'roe_yoy']
		retDict['roe_yearly'] = df.loc[0, 'roe_yearly']
		retDict['roa2_yearly'] = df.loc[0, 'roa2_yearly']
		retDict['grossprofit_margin'] = df.loc[0, 'grossprofit_margin']
		retDict['q_gsprofit_margin'] = df.loc[0, 'q_gsprofit_margin']
		retDict['profitmargin_q'] = df.loc[0, 'q_dtprofit']
		retDict['grossprofitmargin_ttm'] = df.loc[0, 'profit_dedt']

	return retDict

#获取TTM需要的年份和季度
def getTTMYearAndQuarter(cur_datetime, type='ttm'):
	this_year = cur_datetime.year
	quarter  = (cur_datetime.month - 1) //3 + 1
	ret = []
	if type == 'ttm':
		if quarter == 1:
			return [(str(this_year-1)+'1001',str(this_year-1)+'1231', 1)]
		elif quarter == 2:
			return [(str(this_year)+'0101',str(this_year)+'0331', 1), (str(this_year-1)+'1001',str(this_year-1)+'1231', 1), (str(this_year-1)+'0101',str(this_year-1)+'0331', -1)]
		elif quarter == 3:
			return[(str(this_year)+'0330',str(this_year)+'0701', 1), (str(this_year-1)+'1001',str(this_year-1)+'1231', 1), (str(this_year-1)+'0330',str(this_year-1)+'0701', -1)] 
		else :
			return[(str(this_year)+'0701',str(this_year)+'1001', 1), (str(this_year-1)+'1001',str(this_year-1)+'1231', 1), (str(this_year-1)+'0630',str(this_year-1)+'0930', -1)] 
	else:
		if quarter == 1:
			return [(str(this_year-1)+'1001',str(this_year-1)+'1231', 1)]
		elif quarter == 2:
			return [(str(this_year)+'0101',str(this_year)+'0331', 1)]
		elif quarter == 3:
			return[(str(this_year)+'0330',str(this_year)+'0630', 1)] 
		else :
			return[(str(this_year)+'0630',str(this_year)+'0931', 1)] 			
if __name__ == '__main__':
	factor(date.today().replace(year=2018)+timedelta(days=-1), '600230.SH')
