import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
from scipy.ndimage.filters import gaussian_filter1d
from statsmodels.iolib.summary2 import summary_col

plt.rc('legend',fontsize=14)
plt.rc('axes', labelsize=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

def user_accuracy(df, ext):
    y = np.array(df.groupby("rank2")["correct"].sum()/df.groupby("rank2")["correct"].count())[0:50]
    x = range(1,len(y)+1)
    z = df.groupby("rank2")["correct"].count()[0:50]
    lower_bound = y-(1.96/z)*np.sqrt((y*(z-y*z)))
    upper_bound = y+(1.96/z)*np.sqrt((y*(z-y*z)))
    plt.plot(x, 100*y, color='lightgray', linewidth=2, ls="solid")
    plt.errorbar(x, 100*y, yerr=100*(1.96/z)*np.sqrt((y*(z-y*z))), fmt='o', color='black',
                 ecolor='lightgray', elinewidth=4, capsize=0)
    plt.ylabel('Percent Correct')
    plt.xlabel('Image Position')
    plt.ylim(70,99)
    plt.savefig("plots/accuracy_over_exposure_user_{}.svg".format(ext), dpi=300)
    plt.clf()


def image_accuracy(df, ext):
    df["c"] =pd.Categorical.from_array(df.angel).labels
    x = df.groupby("c").correct.count()
    y = 100*df.groupby("c").correct.mean()
    plt.scatter(x, y, color="black")
    plt.ylabel('Percent Correct')
    plt.xlabel('Image Position')
    plt.savefig("plots/accuracy_over_exposure_image_{}.svg".format(ext), dpi=300)
    plt.clf()


def reg_coef(res1, ext):
    x = range(2,12)
    y = []
    lower_bound = []
    upper_bound = []
    for index, i in enumerate(res1.params[0:10]):
        y.append(100*i)
    plt.plot(x, y, color='lightgray', linewidth=2, ls="solid")
    plt.errorbar(x, y, yerr=100*1.96*res1.bse[0:10], fmt='o', color='black',
                 ecolor='lightgray', elinewidth=4, capsize=0)
    plt.xticks(np.arange(2, 11, step=1))
    plt.ylabel('Percent Correct')
    plt.xlabel('Image Position')
    plt.ylim(0,29)
    plt.savefig("plots/regression_coefficients_{}.svg".format(ext), dpi=300)
    plt.clf()


def fourregs(df1,df2,df3,df4, name):
    print(df1.shape)
    print(df2.shape)
    print(df3.shape)
    print(df4.shape)
    if sys.argv[1]=="fe":
        print(name)
        angel1 = pd.get_dummies(df1.angel)
        ip1 = pd.get_dummies(df1.ip)        
        angel1.columns = ["z_a_{}".format(index) for index, i in enumerate(angel1.columns)]
        ip1.columns = ["z_ip_{}".format(index) for index, i in enumerate(ip1.columns)]
        print("1-done")
        angel2 = pd.get_dummies(df2.angel)
        ip2 = pd.get_dummies(df2.ip)
        angel2.columns = ["z_a_{}".format(index) for index, i in enumerate(angel2.columns)]
        ip2.columns = ["z_ip_{}".format(index) for index, i in enumerate(ip2.columns)]
        angel3 = pd.get_dummies(df3.angel)
        ip3 = pd.get_dummies(df3.ip)
        angel3.columns = ["z_a_{}".format(index) for index, i in enumerate(angel3.columns)]
        ip3.columns = ["z_ip_{}".format(index) for index, i in enumerate(ip3.columns)]
        angel4 = pd.get_dummies(df4.angel)
        ip4 = pd.get_dummies(df4.ip)
        angel4.columns = ["z_a_{}".format(index) for index, i in enumerate(angel4.columns)]
        ip4.columns = ["z_ip_{}".format(index) for index, i in enumerate(ip4.columns)]
        print("4-done")
        if sys.argv[3] == "laptop":
            ip1 = ip1[ip1.columns[0:10]]
            ip2 = ip2[ip2.columns[0:10]]
            ip3 = ip3[ip3.columns[0:10]]
            ip4 = ip4[ip4.columns[0:10]]
            angel1 = angel1[angel1.columns[0:10]]
            angel2 = angel2[angel2.columns[0:10]]
            angel3 = angel3[angel3.columns[0:10]]
            angel4 = angel4[angel4.columns[0:10]]        
        df1 = pd.merge(df1,ip1,left_index=True,right_index=True)
        df1 = pd.merge(df1,angel1,left_index=True,right_index=True)
        print("df1: {}".format(df1.shape))
        df2 = pd.merge(df2,ip2,left_index=True,right_index=True)
        df2 = pd.merge(df2,angel2,left_index=True,right_index=True)
        print("df2: {}".format(df2.shape))
        df3 = pd.merge(df3,ip3,left_index=True,right_index=True)
        df3 = pd.merge(df3,angel3,left_index=True,right_index=True)
        print("df3: {}".format(df3.shape))
        df4 = pd.merge(df4,ip4,left_index=True,right_index=True)
        df4 = pd.merge(df4,angel4,left_index=True,right_index=True)
        print("df4: {}".format(df4.shape))
        if name == "cardinal":
            l1 = ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"] + list(ip1.columns)+ list(angel1.columns)
            l2 = ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"] + list(ip2.columns)+ list(angel2.columns)
            l3 = ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"] + list(ip3.columns)+ list(angel3.columns)
            l4 = ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"] + list(ip4.columns)+ list(angel4.columns)
        else: 
            l1 = ["log_rank"] + list(ip1.columns)+ list(angel1.columns)
            l2 = ["log_rank"] + list(ip2.columns)+ list(angel2.columns)
            l3 = ["log_rank"] + list(ip3.columns)+ list(angel3.columns)
            l4 = ["log_rank"] + list(ip4.columns)+ list(angel4.columns)
    else:
        if name == "cardinal":
            l1 = ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"]
            l2 = ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"]
            l3 = ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"]
            l4 = ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"]
        else: 
            l1 = ["log_rank"]
            l2 = ["log_rank"]
            l3 = ["log_rank"]
            l4 = ["log_rank"]
   
    res1 = sm.OLS(df1['correct'], sm.add_constant(df1[l1]), M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': df1.angel})
    print("res1-run finished")
    res2 = sm.OLS(df2['correct'], sm.add_constant(df2[l2]), M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': df2.angel})
    print("res2-run finished")
    res3 = sm.OLS(df3['correct'], sm.add_constant(df3[l3]), M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': df3.angel})
    print("res3-run finished")
    res4 = sm.OLS(df4['correct'], sm.add_constant(df4[l4]), M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': df4.angel})
    print("res4-run finished")

    res_df = pd.DataFrame({"coefs":res1.params, "se":res1.bse})
    res_df = pd.DataFrame({"coefs":res2.params, "se":res2.bse})
    res_df = pd.DataFrame({"coefs":res3.params, "se":res3.bse})
    res_df = pd.DataFrame({"coefs":res4.params, "se":res4.bse})
    res_df.to_csv("{}_1.csv".format(name))
    res_df.to_csv("{}_2.csv".format(name))
    res_df.to_csv("{}_3.csv".format(name))
    res_df.to_csv("{}_4.csv".format(name))

    dfoutput = summary_col([res1, res2, res3, res4],stars=True,
                           info_dict={
                            'N':lambda x: "{0:d}".format(int(x.nobs)),
                            'R2':lambda x: "{:.2f}".format(x.rsquared)},
                            model_names=["(1)","(2)","(3)","(4)","(5)"])
    print(dfoutput)
    f = open('regression/{}.txt'.format(name), 'w+')
    # f.write(dfoutput.as_latex())
    f.close()
    if "card" in name:
        reg_coef(res1, "1")
        reg_coef(res2, "2")
        reg_coef(res3, "3")
        reg_coef(res4, "4")

def g(df):
    return np.array(df.groupby("rank2")["correct"].sum()/df.groupby("rank2")["correct"].count())[0:10], df.groupby("rank2")["correct"].count()[0:10]

def splitter(df, split, v1, v2):
        return df[df[split]==v1], df[df[split]==v2]

def dummy(df):
    ip = pd.get_dummies(df.ip)
    angel = pd.get_dummies(df.angel)
    angel.columns = ["z_a_{}".format(index) for index, i in enumerate(angel.columns)]
    ip.columns = ["z_ip_{}".format(index) for index, i in enumerate(ip.columns)]
    if sys.argv[3] == "laptop":
        angel = angel[angel.columns[0:10]]
        ip = ip[ip.columns[0:10]]
    else:
        angel = angel[angel.columns]
        ip = ip[ip.columns]
    return ip, angel

def merger(df, ip, angel):
    df = pd.merge(df,ip,left_index=True,right_index=True)
    df = pd.merge(df,angel,left_index=True,right_index=True)
    print(df.shape)
    return df

def regress(data, ip, angel, split, side):
    if (split=="first_correct") and (side=="b"):
        l = ["3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"]
        res = sm.OLS(data[data.ranking!=1]['correct'], data[data.ranking!=1][l+list(ip.columns)+list(angel.columns)], M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': data[data.ranking!=1].angel})
    else:
        l = ["2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"]
        try:
            res = sm.OLS(data['correct'], data[l+list(ip.columns)+list(angel.columns)], M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': data.angel})
        except:
            print("bug_1")
            fix = pd.DataFrame(data.groupby("angel").correct.count()).reset_index()
            fix.columns = ["angel","fix"]
            data = pd.merge(data, fix, on="angel")
            res = sm.OLS(data['correct'], data[l+list(angel.columns)], M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': data.angel})
            # res = sm.OLS(data['correct'], sm.add_constant(data[l+list(angel.columns)]), M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': data.angel})
            print(res)
            # except:
            #     res = sm.OLS(data['correct'], data[l+list(ip.columns)+list(angel.columns)], M=sm.robust.norms.HuberT()).fit()

    df = pd.DataFrame({"coefs":res.params, "se":res.bse})
    df = df.reset_index()
    df.to_csv("regression/{}_{}.csv".format(split, side))
    return df

def hetero(data, split, v1, v2, l1, l2):
    data_a, data_b = splitter(data, split, v1, v2)
    ip_a, angel_a = dummy(data_a)
    ip_b, angel_b = dummy(data_b)
    data_a = merger(data_a, ip_a, angel_a)
    data_b = merger(data_b, ip_b, angel_b)
    a = regress(data_a, ip_a, angel_a, split, "a")
    b = regress(data_b, ip_b, angel_b, split, "b")
    a = a[0:10]
    if split=="first_correct":
        b = b[0:9]
    else:
        b = b[0:10]
    y1 = [0]
    y2 = [0]
    z1 = [0]
    z2 = [0]
    for index, i in enumerate(a.coefs[0:10]):
        y1.append(i)
        z1.append(a.se[index])
    for index, i in enumerate(b.coefs[0:10]):
        y2.append(i)
        z2.append(b.se[index])
    x1 = range(1,len(y1)+1)
    x2 = range(1,len(y2)+1)
    if split=="first_correct":
        x2 = range(2,len(y2)+2)
    plt.errorbar(x1, (100*np.array(y1)), yerr=list(100*(1.96*np.array(z1))), color='red', ls="solid", linewidth=4, alpha=.5, ecolor='red', elinewidth=6, capsize=0, fmt="", label=l1)
    plt.errorbar(x2, (100*np.array(y2)), yerr=list(100*(1.96*np.array(z2))), color='blue', ls="solid", linewidth=4, alpha=.5, ecolor='blue', elinewidth=6, capsize=0, fmt="", label=l2)
    plt.ylabel('Percentage Point Change')
    plt.xlabel('Image Position')
    plt.xticks(np.arange(1, 11, step=1))
    # Specify Y Limit
    if split=="first_correct":
        plt.ylim(-22,3)
        yint = range(-20, 3, 4)
        plt.yticks(yint)
    elif split=="hq":
        plt.ylim(-3,16)
    elif split=="has_person":
        plt.ylim(-3,22)
    elif split=="mobile":
        plt.ylim(-3,22)
    elif split=="mask_1st_4th":
        plt.ylim(-3,16)
    elif split=="time_1st_4th":
        plt.ylim(-3,16)
    elif split=="one_object_disappeared":
        plt.ylim(-3,15)
    elif split=="entropy_1st_4th":
        plt.ylim(-2,17)
    elif split=="coco_left":
        plt.ylim(-3,15)
    elif split=="accuracy_1st_4th":
        plt.ylim(-3,15)
    plt.legend(frameon=False, loc="lower right")
    leg = plt.legend(loc="lower right")
    leg.get_frame().set_linewidth(0.0)
    plt.savefig("plots/hetero_{}.svg".format(split), dpi=300)
    plt.clf()


def hetero_regs(df2,h):
    print(df2.shape)
    df2 = df2[df2.ranking<=10]
    print(df2.shape)
    ip2, angel2 = dummy(df2)
    if sys.argv[1]=="fe":
        angel2 = pd.get_dummies(df2.angel)
        ip2 = pd.get_dummies(df2.ip)
        angel2.columns = ["z_a_{}".format(index) for index, i in enumerate(angel2.columns)]
        ip2.columns = ["z_ip_{}".format(index) for index, i in enumerate(ip2.columns)]
    elif sys.argv[3] == "laptop":
        ip2 = ip2[ip2.columns[0:10]] 
        angel2 = angel2[angel2.columns[0:10]]          
    df2 = pd.merge(df2,ip2,left_index=True,right_index=True)
    df2 = pd.merge(df2,angel2,left_index=True,right_index=True)
    walri = []
    for index, i in enumerate(h):
        
        i = i[0]
        print(i)
        xx = df2.copy()
        if i=="first_correct":
            xx.loc[df.first_correct == 2, 'first_correct'] = 0
            xx = xx[xx.first_correct<=1]
            xx = xx[xx.ranking>1]
        xx = xx[(xx[i]==1)|(xx[i]==0)]
        xx["log_rank_x_{}".format(i)] = xx.log_rank*df2[i].astype(float)
        l2 = ["log_rank", i, "log_rank_x_{}".format(i)] +  list(angel2.columns) # +list(ip2.columns)
        xx[l2 + ["correct"]].to_csv("r_{}.csv".format(i))
        res2 = sm.OLS(xx['correct'], sm.add_constant(xx[l2].astype(float)), M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': xx.angel})
        # res2 = sm.OLS(xx['correct'], sm.add_constant(xx[l2].astype(float))).fit()
        print("res2-run finished")
        res_df = pd.DataFrame({"coefs":res2.params, "se":res2.bse})
        res_df.to_csv("h_{}_reg.csv".format(i))
        walri.append(res2)
    xx = df2.copy()
    xx.loc[df.first_correct == 2, 'first_correct'] = 0
    xx = xx[xx.first_correct<=1]
    xx = xx[xx.ranking>1] 
    h_1 = [i[0] for i in h if i !="first_correct"]
    for i in h_1:
        df2["log_rank_x_{}".format(i)] = df2.log_rank*df2[i].astype(float)
    h_2 = [i[0] for i in h]
    # for i in h_2:
    #     xx["log_rank_x_{}".format(i)] = xx.log_rank*xx[i].astype(float)
    h_interaction_1 = ["log_rank_x_{}".format(i[0]) for i in h if i[0] !="first_correct"]
    # h_interaction_2 = ["log_rank_x_{}".format(i[0]) for i in h]
    l4 = h_1+h_interaction_1 + ["log_rank"] #+list(angel2.columns)
    # l5 = h_2+h_interaction_2+list(angel2.columns)
    print(l4)
    # print(l5)

    df2[["correct"] + l4].astype(float).to_csv("~/Desktop/go.csv")
    res2 = sm.OLS(df2['correct'], sm.add_constant(df2[l4].astype(float)), M=sm.robust.norms.HuberT()).fit()
    walri.append(res2)
    res2 = sm.OLS(xx['correct'], sm.add_constant(xx[l5].astype(float)), M=sm.robust.norms.HuberT()).fit(cov_type='cluster', cov_kwds={'groups': xx.angel})
    walri.append(res2)
    dfoutput = summary_col(walri,stars=True,
                           info_dict={
                            'N':lambda x: "{0:d}".format(int(x.nobs)),
                            'R2':lambda x: "{:.2f}".format(x.rsquared)},
                            model_names=["(1)","(2)","(3)","(4)","(5)","(6)","(7)","(8)","(9)","(10)", "(11)","(12)"],
                            regressor_order=[
                                "log_rank",
                                "log_rank_x_hq", "hq",
                                "log_rank_x_accuracy_1st_4th", "accuracy_1st_4th",
                                "log_rank_x_mask_1st_4th", "mask_1st_4th",
                                "log_rank_x_entropy_1st_4th", "entropy_1st_4th",
                                "log_rank_x_one_object_disappeared", "one_object_disappeared",
                                "log_rank_x_first_correct", "first_correct",
                                "log_rank_x_has_person", "has_person",
                                "log_rank_x_time_1st_4th", "time_1st_4th",
                                "log_rank_x_mobile", "mobile",
                                "log_rank_x_coco_left","coco_left"])
    print(dfoutput)
    f = open('regression/heterogeneous.txt', 'w+')
    f.write(dfoutput.as_latex())
    f.close()


if __name__ == "__main__":
    print("specify (1) fe or not (2) all or cardinal (3) laptop or not")
    df = pd.read_csv("tabular-data/deepangel.csv")
    hq = pd.read_csv("tabular-data/subjective_quality_ranking.csv")
    hq = hq[["angel","original","image_id","object","rating"]]
    df.time_stamp = pd.to_datetime(df.time_stamp)
    gf = df.groupby('ip')['time_stamp'].rank(ascending=True)
    df["rank2"] = gf
    vc = pd.DataFrame(df.ip.value_counts())
    vc["index"]=vc.index
    vc.columns = ["total", "ip"]
    df = pd.merge(df,vc,left_on="ip",right_on="ip")
    df = df[["ip","correct","time_stamp","rank2","total","platform","browser",'imageone','imagetwo']]
    df["correct"] = (df.correct=="correct").astype(int)
    df["angel"]=df.imageone
    df["coco"] = df.imageone
    df["coco_left"] = [int("val2017" in i) for i in df.imageone]
    df.loc[df.coco_left == 1, 'angel'] = df["imagetwo"]
    df.loc[df.coco_left == 0, 'coco'] = df["imagetwo"]
    df.coco = df.coco.str.replace('https:/', 'https://')
    angel = pd.get_dummies(df.angel)
    coco = pd.get_dummies(df.coco)
    platform = pd.get_dummies(df.platform)
    browser = pd.get_dummies(df.browser)
    ip = pd.get_dummies(df.ip)
    angel.columns = ["z_a_{}".format(index) for index, i in enumerate(angel.columns)]
    coco.columns = ["z_coco_{}".format(index) for index, i in enumerate(coco.columns)]
    platform.columns = ["z_p_{}".format(i) for i in platform.columns]
    browser.columns = ["z_b_{}".format(i) for i in browser.columns]
    ip.columns = ["z_ip_{}".format(index) for index, i in enumerate(ip.columns)]
    print("Beginning OHO merge")
    df["morethanten"]=(df.rank2>10).astype(int)
    df['ranking'] = df.rank2.copy()
    df.loc[df.ranking>11,"ranking"] = 11
    df['log_rank']=np.log(df.rank2)
    rank_oho = pd.get_dummies(df.ranking)
    rank_oho.columns = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "More than 10"]
    df = pd.merge(df,rank_oho, left_index=True, right_index=True)
    df = df.reset_index(drop=True)
    lol = pd.DataFrame(df.groupby(["ip","correct"]).rank2.min()).reset_index()
    lol = lol[lol.correct==1]
    lol = lol[["ip","rank2"]]
    lol.columns = ["ip","first_correct"]
    df = pd.merge(df,lol, on="ip", how="left")
    df = pd.merge(df,hq, on="angel",how="left")
    high_quality = list(hq[hq.rating<=3].angel)
    df["hq"] = df.angel.isin(high_quality)
    df1 = df[df.image_id.notnull()]
    df1 = df1[df1.rating!=9]
    df2 = df1[df1.total>=10]
    temp = df2[["ip","angel","ranking"]]
    temp = temp.sort_values(["ip","ranking"])
    temp = temp[["ip","angel"]].drop_duplicates(keep="first")
    temp["not_dup"] = 1
    del temp["ip"]
    del temp["angel"]
    df3 = pd.merge(df2,temp,left_index=True, right_index=True, how="right")
    df4 = df3[df3.angel.isin(high_quality)]
    print(df.shape)
    print(df1.shape)
    print(df2.shape)
    print(df3.shape)
    print(df4.shape)
    user_accuracy(df,"1")
    user_accuracy(df2,"2")
    user_accuracy(df3,"3")
    user_accuracy(df4,"4")
    for i in [df,df2,df3,df4]:
        print("1st image: {}".format(round(i[i.rank2==1].correct.mean(),3)))
        print("10th image: {}".format(round(i[i.rank2==10].correct.mean(),3)))
    if sys.argv[2] == "all":
        # fourregs(df1,df2,df3,df4,lin_coef, "linear_kink")
        fourregs(df,df2,df3,df4,"log")
        fourregs(df,df2,df3,df4,"cardinal")
    elif sys.argv[2] == "cardinal":
        fourregs(df,df2,df3,df4,"cardinal")

    print("HETEROGENEOUS ANALYSIS")
    ic = pd.read_csv("tabular-data/image_characteristics.csv")
    df2 = pd.merge(df2,ic,how="left",on="image_id")
    df2["has_person"] = df2.object=="person"
    df2["mobile"] = 0
    df2["one_object_disappeared"] = df2.objects_disappeared==1
    df2.loc[(df2.platform=="iphone") | (df2.platform=="android"), "mobile"]=1

    xxx = pd.DataFrame(df2.groupby("angel")[["p_masked","correct", "entropy"]].mean())
    xxx = xxx.reset_index()
    xxx.columns = ["angel","img_mask","img_acc", "img_entropy"]
    df2 = pd.merge(df2,xxx, on="angel")
    df2["accuracy_1st_4th"] = -1
    df2.loc[df2.img_acc>df2.groupby("angel").correct.mean().describe()[6], "accuracy_1st_4th"]=0
    df2.loc[df2.img_acc<df2.groupby("angel").correct.mean().describe()[4], "accuracy_1st_4th"]=1

    df2["mask_1st_4th"] = -1
    df2.loc[df2.img_mask>df2.groupby("angel").p_masked.mean().describe()[6], "mask_1st_4th"]=0
    df2.loc[df2.img_mask<df2.groupby("angel").p_masked.mean().describe()[4], "mask_1st_4th"]=1

    df2["entropy_1st_4th"] = -1
    df2.loc[df2.img_entropy>df2.groupby("angel").entropy.mean().describe()[6], "entropy_1st_4th"]=0
    df2.loc[df2.img_entropy<df2.groupby("angel").entropy.mean().describe()[4], "entropy_1st_4th"]=1

    xxx = pd.DataFrame(df2[df.ranking==1].groupby("angel")[["correct"]].mean())
    xxx = xxx.reset_index()
    xxx.columns = ["angel","img_acc_1"]
    df2 = pd.merge(df2,xxx, on="angel")
    df2["accuracy_1_1st_4th"] = -1
    df2.loc[df2.img_acc_1>df2.groupby("angel").correct.mean().describe()[6], "accuracy_1_1st_4th"]=0
    df2.loc[df2.img_acc_1<df2.groupby("angel").correct.mean().describe()[4], "accuracy_1_1st_4th"]=1

    xxx = pd.DataFrame(df2[df.ranking==10].groupby("angel")[["correct"]].mean())
    xxx = xxx.reset_index()
    xxx.columns = ["angel","img_acc_10"]
    df2 = pd.merge(df2,xxx, on="angel")
    df2["accuracy_10_1st_4th"] = -1
    df2.loc[df2.img_acc_10>df2.groupby("angel").correct.mean().describe()[6], "accuracy_10_1st_4th"]=0
    df2.loc[df2.img_acc_10<df2.groupby("angel").correct.mean().describe()[4], "accuracy_10_1st_4th"]=1


    mint = pd.DataFrame(df2[df2.rank2==1].groupby('ip').time_stamp.min())
    maxt = pd.DataFrame(df2[df2.rank2==10].groupby('ip').time_stamp.min())
    tt = pd.merge(mint, maxt, left_index=True,right_index=True)
    tt = tt.reset_index()
    tt["difference"] = tt.time_stamp_y-tt.time_stamp_x
    tt["seconds"] = tt.difference.dt.seconds
    tt = tt[["ip","seconds"]]
    df2 = pd.merge(df2,tt, on="ip")
    df2["time_1st_4th"] = -1
    df2.loc[df2.seconds>df2.groupby("ip").seconds.mean().describe()[6], "time_1st_4th"]=0
    df2.loc[df2.seconds<df2.groupby("ip").seconds.mean().describe()[4], "time_1st_4th"]=1

    
    high_quality = ["hq", 1, 0, "High Quality Rating", "Low Quality Rating"]
    first_correct = ["first_correct", 1, 2, "First Guess Correct", "First Guess Incorrect"]
    has_person = ["has_person", 1, 0, "Person Disappeared", "Something Else Disappeared"]
    angel_right = ["coco_left", 1, 0, "Right", "Left"]
    mobile = ["mobile", 1, 0, "Mobile", "Computer"]
    img_acc_1st_4th = ["accuracy_1st_4th", 1, 0, "Low Accuracy", "High Accuracy"]
    img_acc_1_1st_4th = ["accuracy_1_1st_4th", 1, 0, "Low Accuracy", "High Accuracy"]
    img_acc_10_1st_4th = ["accuracy_10_1st_4th", 1, 0, "Low Accuracy", "High Accuracy"]
    mask_1st_4th = ["mask_1st_4th", 1, 0, "Small Mask", "Large Mask"]
    time_1st_4th = ["time_1st_4th", 1, 0, "Fast Completion Time", "Slow Completion Time"]
    entropy_1st_4th = ["entropy_1st_4th", 1, 0, "Low Entropy", "High Entropy"]
    one_object = ["one_object_disappeared", 1, 0, "One Object Disappeared", "More than One Object Disappeared"]
    # img_acc_10_1st_4th, img_acc_1_1st_4th, 
    splits = [entropy_1st_4th, angel_right, img_acc_1st_4th, high_quality, first_correct, has_person, mobile, mask_1st_4th, time_1st_4th, one_object]

    splits = [high_quality, img_acc_1st_4th, mask_1st_4th, entropy_1st_4th, one_object, first_correct, has_person, time_1st_4th, mobile,   angel_right   ]
    # splits = [first_correct]
    print("Begining Log Rank Hetero Regs")
    hetero_regs(df2, splits)


    for split in splits:
        # df2[split[0]] = df2[split[0]].astype(int)
        print("start plotting {}".format(split[0]))
        hetero(data=df2, split=split[0], v1=split[1], v2=split[2], l1=split[3], l2=split[4])
        print("finished plotting {}".format(split[0]))

 #    w = df2[df2.hq==1].groupby("angel")[["entropy","correct", "p_masked"]].mean().reset_index()
 #    plt.scatter(w.entropy,w.correct)
 #    plt.xlabel('Entropy')
 #    plt.ylabel('Accuracy')
 #    plt.savefig("plots/z_entropy.png", dpi=300)
 #    plt.clf()

 #    plt.scatter(w.p_masked,w.correct)
 #    plt.xlabel('Proportion Masked')
 #    plt.ylabel('Accuracy')
 #    plt.savefig("plots/z_p_masked.png", dpi=300)
 #    plt.clf()


    

 #    w2 = df2.groupby(["image_id", 'mobile', "hq", "mask_1st_4th","entropy_1st_4th"])["correct"].mean()
 #    w2 = w2.reset_index()
 #    plt.hist(w2[w2.mobile==0].correct, bins=20, alpha=.4, color="blue", label="Computer")
 #    plt.hist(w2[w2.mobile==1].correct, bins=20, alpha=.4, color="red", label="Mobile")
 #    plt.ylabel('Frequency')
 #    plt.xlabel('Accuracy')
 #    plt.legend()
 #    # plt.title("Accuracy Histogram of Computer vs. Mobile")
 #    plt.savefig("plots/z_hist_comp_mobile.png", dpi=300)
 #    plt.clf()
 #    plt.hist(w2[w2.hq==1].correct, bins=20, alpha=.4, color="blue", label="High Quality")
 #    plt.hist(w2[w2.hq==0].correct, bins=20, alpha=.4, color="red", label="Low Quality")

 #    plt.ylabel('Frequency')
 #    plt.xlabel('Accuracy')
 #    plt.legend()
 #    # plt.title("Histogram Accuracy (High Quality)")
 #    plt.savefig("plots/z_hist_quality.png", dpi=300)
 #    plt.clf()
 #    plt.hist(w2[w2.mask_1st_4th==1].correct, bins=20, alpha=.4, color="blue", label="Large Mask")
 #    plt.hist(w2[w2.mask_1st_4th==0].correct, bins=20, alpha=.4, color="red", label="Small Mask")

 #    plt.ylabel('Frequency')
 #    plt.xlabel('Accuracy')
 #    plt.legend()
 #    # plt.title("Histogram Accuracy (High Quality)")
 #    plt.savefig("plots/z_hist_mask.png", dpi=300)
 #    plt.clf()
 #    plt.hist(w2[w2.entropy_1st_4th==1].correct, bins=20, alpha=.4, color="blue", label="High Entropy")
 #    plt.hist(w2[w2.entropy_1st_4th==0].correct, bins=20, alpha=.4, color="red", label="Low Entropy")

 #    plt.ylabel('Frequency')
 #    plt.xlabel('Accuracy')
 #    plt.legend()
 #    # plt.title("Histogram Accuracy (High Quality)")
 #    plt.savefig("plots/z_hist_entropy.png", dpi=300)
 #    plt.clf()

 # #    res1 = pd.read_csv("cardinal_2.csv")
 # #    ext = 2
 # #    x = range(1,12)
 # #    y = [0]
 # #    z = [0]
 # #    lower_bound = []
 # #    upper_bound = []
 # #    for index, i in enumerate(res1.coefs[0:10]):
 # #        y.append(100*i)
 # #        z.append(res1.se[index])
 # #    print(len(x))
 # #    print(len(y))
 # #    print(len(z))
 # #    plt.plot(x, y, color='lightgray', linewidth=2, ls="solid")
 # #    plt.errorbar(x, y, yerr=100*1.96*np.array(z), fmt='o', color='black',
 # #                 ecolor='lightgray', elinewidth=4, capsize=0)
 # #    plt.xticks(np.arange(1, 11, step=1))
 # #    plt.ylabel('Percentage Point Improvement')
 # #    plt.xlabel('Image Position')
 # #    plt.ylim(-2,15)
 # #    plt.savefig("plots/regression_coefficients_{}.png".format(ext), dpi=300)
 # #    plt.clf()
    
    ext = "yo"
    walrus = pd.DataFrame(df1.groupby("rank2").correct.count()).reset_index()
    plt.rc('ytick', labelsize=8)
    plt.bar(range(1,51),walrus.correct[0:50], width=.5, color="black")
    plt.ylabel("Number of Individuals")
    plt.xlabel("Image Position")
    plt.savefig("plots/individuals_by_position.png".format(ext), dpi=300)
    plt.clf()
    plt.rc('ytick', labelsize=12)
    plt.hist(100*(df1.groupby("angel").correct.mean()),bins=50,normed=True, color="black")
    plt.xlabel("Identification Accuracy of Images")
    plt.ylabel("Proportion of Images")
    plt.savefig("plots/accuracy_histogram_image.png".format(ext), dpi=300, color="black")
    plt.clf()

 # #    plt.hist(100*(df1.groupby("ip").correct.mean()),bins=50,normed=True)
 # #    plt.ylabel("Identification Accuracy")
 # #    plt.xlabel("Proportion")
 # #    plt.savefig("plots/accuracy_histogram_ip.png".format(ext), dpi=300, color="black")
 # #    plt.clf()
    
 # #    # y2, z2 = g(df[(df.rating==9)])
 # #    # x = range(1,len(y2)+1)
 # #    # plt.errorbar(x, 100*y2, yerr=list(100*(1.96/z2)*np.sqrt((y2*(z2-y2*z2)))), color='black', linewidth=4, ls="dotted",  alpha=.8, ecolor='darkgray', elinewidth=4, capsize=0, fmt="")
 # #    # plt.ylim(40,69)
 # #    # plt.ylabel('Percent Correct')
 # #    # plt.xlabel('Image Position')
 # #    # plt.xticks(np.arange(0, 11, step=1))
 # #    # plt.savefig("plots/control_d.png", dpi=300)
 # #    # plt.clf()
 # #    # print(df1.shape)
 # #    # print(df1.angel.nunique())
 # #    # print(df1.coco.nunique())
 # #    # print("done")

    # dfx = df2[df2.rank2<11]
    # mm = pd.DataFrame(dfx.groupby("ip").rank2.count())
    # mm["ip"] = mm.index
    # mm.columns = ["cc","ip"]
    # dfx = pd.merge(dfx,mm,on="ip")
    # dfx.shape
    # dfx = dfx[dfx.cc==10]
    # dfx["seconds"]=dfx.td.dt.total_seconds()
    # dfx["sec_diff"] = dfx.sort_values(["ip","rank2"]).groupby("ip").seconds.diff()
    # dfx[["ip","rank2","seconds","sec_diff"]].sort_values(["ip","rank2"]).head(10)
    # mm = pd.DataFrame(dfx.groupby("ip").correct.expanding().mean())
    # # mm["ip"] = mm.index
    # mm = mm.reset_index()
    # mm.columns = ['ip','level_1', 'ra']
    # # # dfx = pd.merge(dfx,mm,on="ip")
    # mm["rank2"] = mm.groupby("ip")["level_1"].rank()

    # dfx["lag_seconds"] = dfx.seconds.shift(1)
    # dfx["previous_guess_incorrect"] = dfx["shift"]*-1+1
    # dfx["seconds_X_previous_guess_incorrect"] = dfx["previous_guess_incorrect"]*dfx["seconds"]

