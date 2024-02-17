# MIDEAST_FINAL
**Dataset (MIDEAST-TE)_v2.0_deduplicated:**
<table border="1" width="500px">
<tr >
  <th align="center">#atomic events</th>
  <th align="center">#CE</th>
  <th align="center">#CE-max/min</th>
</tr>
<tr >
  <td bgcolor="#CCFFF"><b>8623</b></td>
  <td>120</td>
  <td>112 / 19</td>
</tr>
<tr>
  <th align="center">#RelationType</th>
  <th align="center">#Entities</th>
  <th align="center">#Docs(MD5_list)</th>
</tr>
<tr>
  <td>20</th>
  <td bgcolor="#CCFFF"><b>896</b> / 2794</td>
  <td bgcolor="#CCFFF"><b>4826</b></td>
</tr>
</table>


> 重复文章数量：1029

## 文件结构说明
### 1. all_events.csv:

>GPT 重新抽取结果

### 2. all_events_final.csv

>all_evnets去重结果
>
>(根据三元组，timid及ce_id去重) ['Subject', 'Relation_id', 'Object','timid','ce_id']

### 3. MIDEAST_Clean_final_v2.csv（参考）:

MIDEAST 原始数据集（summary+contriver去重后的结果）

> 字段说明：
>
> **atomic_events**：*ce周期下atomic_events数量*
>
> **timespan**：*ce时间周期*
>
> **timdiff**: *与上一个atomic_event的间隔时间*
>
> **relative_time**: *atomic_event发生的相对时间*
>
> **is_Uniform**: *ce分布是否均匀，均匀为True*
>
> **is_drop**：*Md5是否重复，重复为True*

### 4. ceid2Md5.json:

每个ce中包含Md5

### 5. Md52timid.json

每个Md5对应的timid

### 6. date2id.txt

date对应的timid

### 7. summary_list_clean.json

Md5对应的summary

> vicuna-13b-v1.5-16k
>
> --temperature 1.0 --max-new-tokens 512  --repetition_penalty 1.0 --load_8bit False

### 8. doc_clean_ALL.json

Md5对应的原始文章数据