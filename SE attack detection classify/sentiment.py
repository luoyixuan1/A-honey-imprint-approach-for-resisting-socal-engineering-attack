import textblob
import matplotlib.pyplot as plt
import pandas as pd

file = pd.read_csv("dataset.csv")
data = file["text"]

polaritys = []
subjectivity = []
sp, sn, op, on = (0, 0, 0, 0)
s, p, n, o, m = (0, 0, 0, 0, 0)
for item in data:
    if len(item) > 1:
        blob = textblob.TextBlob(item)
        x = blob.sentiment.polarity
        y = blob.sentiment.subjectivity
        polaritys.append(x)
        subjectivity.append(y)

        if x > 0:
            if y > 0.5:
                sp += 1
            elif y < 0.5:
                op += 1
            else:
                p += 1
        elif x < 0:
            if y > 0.5:
                sn += 1
            elif y < 0.5:
                on += 1
            else:
                n += 1
        else:
            if y > 0.5:
                s += 1
            elif y < 0.5:
                o += 1
            else:
                m += 1

result_data = {'text':data, 'polaritys': polaritys, 'subjectivity': subjectivity}
df = pd.DataFrame(result_data)
df.to_csv('sentiment_analysis.csv')

data_n = len(polaritys)
print(f'sp:{sp/data_n}, sn:{sn/data_n}, op:{op/data_n}, on:{on/data_n}')
print(f's:{s/data_n}, o:{o/data_n}, p:{p/data_n}, n:{n/data_n}, m:{m/data_n}')
print(sp + sn + op + on + s + p + o + n + m)

plt.scatter(polaritys, subjectivity)    # 输入散点数据
plt.axvline(0, color='black', ls='--', lw='0.5')
plt.axhline(0.5, color='black', ls='--', lw='0.5')
plt.show()          # 显示散点图

