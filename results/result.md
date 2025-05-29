# Part A

> see data in `results/A` folder

plot

- train loss
- valid loss
- train perplexity
- valid perplexity

# Part B

## Perplexity on WikiText-2

transformer: 346.8285

new transformer: 335.8599

lstm: 390.0218

new lstm: 432.4815

rnn: 411.1751

## 2.3.1

Start with `a`

transformer:

```
a <unk> from an <unk> <unk> in <unk> and <unk> <eos> the <unk> <unk> <eos> in mr. baker said there was in <unk> but <unk> <eos> if <unk> there is n't a lot of us here with me for some people just have for the <unk> of them to read and he 's happy <eos> he 's that 's that we should take in a <unk> <eos> he is n't sure that it 's a small time in a thing that 's not <unk> a lot of <unk> and a <unk> says <unk> <unk> a <unk> university veteran of texas <eos>
```

new transformer:

```
speech from robert <unk> <unk> in <unk> calif. is conducting a <unk> <unk> <eos> in mr. <unk> said there was in a new <unk> <eos> if <unk> there is n't a lot of us here mr. <unk> says <eos> the fact that for the past three weeks that he has tried to sell N shares <eos> that 's that 's more than in order and just just a bit smaller from having <unk> <eos> he has done a lot a thing of <unk> a lot of <unk> at this point says <unk> <unk> a <unk> university veteran of texas <eos>
```

lstm:

```
a <unk> from all <unk> <unk> in <unk> and <unk> <eos> but the fact <eos> in mr. month said there was in <unk> where <unk> <eos> if <unk> is out and they really was not seen with them for some <unk> <unk> but for the <unk> of a <unk> <eos> he is <unk> <eos> we 'm still that it does n't be a very step and <eos> he is n't sure any <unk> <unk> is too less in a <unk> that 's not <unk> a lot of <unk> and one <unk> says <unk> <unk> a <unk> <eos> as that many other
```

new lstm:

```
<unk> from all <unk> <unk> in <unk> and <unk> <eos> but the fact that in mr. krenz said there was in <unk> where <unk> <eos> if <unk> there <unk> and they really 'll not have with me for some <unk> <unk> but for the <unk> of them to read and he 's a <unk> that 's that it does n't be a little <eos> and we can believe that 's not it <unk> <eos> he has done a man that 's not <unk> a lot of <unk> and a <unk> says <unk> <unk> a <unk> <eos> as that this is
```

rnn:

```
a <unk> from all <unk> <unk> in <unk> and <unk> <eos> but there would be in mr. krenz for his <unk> in <unk> where <unk> <eos> if <unk> is a <unk> <eos> and the other hand took me for some <unk> <unk> but for the <unk> of homelessness that he says he 's a <unk> <unk> <eos> that 's that we should take in order to turn off <eos> in fact that it 's a small course in a month that 's not <unk> a lot of <unk> and one <unk> says <unk> <unk> a <unk> n.c. that would be the
```

## 2.3.2

Start with `the meaning of life is`

transformer:

```
the meaning of life is <unk> from all <unk> <unk> with <unk> and <unk> <unk> <eos> the <unk> <eos> in mr. gelbart said there is in <unk> where <unk> <eos> if <unk> but what and they really want us to live together for some <unk> <unk> but for the <unk> of whom happened <eos> he also <unk> the <unk> <unk> <eos> that 's that we should be in a <unk> <eos> mr. gelbart 's <unk> from his <unk> <unk> he has done a man a <unk> <unk> of a <unk> <eos> there 's a <unk> says <unk> <unk> <unk> <unk> the most <unk> of the
```

new transformer:

```
extremely interesting but not good for <unk> and <unk> <eos> but there are more time to be a black activist in <unk> where <unk> <eos> if <unk> you want and they really want to have you you for somebody <eos> <unk> <unk> for national football <eos> in the third quarter he agreed to sell them <eos> that 's that 's more than seven others and have just <unk> <eos> if any <unk> <unk> is he has a way to make it more <unk> a lot of <unk> and one <unk> says <unk> <unk> a <unk> university sociologist who runs the <unk> <eos>
```

lstm:

```
the meaning of life is <unk> from all <unk> after their <unk> and <unk> <eos> but the fact <eos> in mr. baker said there was in <unk> where <unk> <eos> if <unk> is out and had just the right to do them for some <unk> <unk> but for the <unk> of a democratic thing <eos> he 's a <unk> that it makes the work <eos> she had in a matter <eos> mr. <unk> is a first <unk> <unk> and he has a consultant and a <unk> of <unk> a <unk> <eos> there say he 's to <unk> the house <unk> <eos> as that many other
```

new lstm:

```
<unk> from all <unk> <unk> in <unk> and <unk> <eos> but the fact that in mr. krenz said there was in <unk> where <unk> <eos> if <unk> there <unk> and they really 'll not have with me for some <unk> <unk> but for the <unk> of them to read and he 's a <unk> that 's that it does n't be a little <eos> and we can believe that 's not it <unk> <eos> he has done a man that 's not <unk> a lot of <unk> and a <unk> says <unk> <unk> a <unk> <eos> as that this is
```

rnn:

```
the meaning of life is <unk> from all of their lives <unk> and <unk> <eos> but there would be in mr. krenz for his <unk> in <unk> where <unk> <eos> if <unk> is a <unk> <eos> and the other hand took me for some <unk> <unk> but for the <unk> of homelessness that he says he 's a <unk> <unk> <eos> that 's that we should take in order to turn off <eos> in fact that it 's a small course in a month that 's not <unk> a lot of <unk> and one <unk> says <unk> <unk> a <unk> n.c. that would be the
```

# Part C

Finetune GPT-2 small on the pubmed_qa dataset.

## Perplexity

Before finetune: 28.32

After finetune: 8.78

## Example

prompt:

```
Question: What causes COVID-19?
Context: COVID-19 is caused by the SARS-CoV-2 virus.
Answer:
```

Before finetune:

```
The SARS-CoV-2 virus is a virus that is present in the human body. It is a virus that is present in the human body.
```

After finetune:

```
The aim of this study was to determine the cause of COVID-19 in a population of patients with severe respiratory syndrome (SARS).
METHODS: A retrospective cohort study was conducted in the Netherlands. Patients with severe SARS were included in the study. Patients with severe SARS were excluded from the study because of the presence of SARS-CoV-2 virus.
RESULTS: The mean age of the patients was 43.5 years (range, 30-54 years).
```

# RLHF

## ORPO

Prompt: A sunset over the mountains
Output: A sunset over the mountains, golden light casting long shadows on the forest floor. A small group of people walking in the forest

Prompt: The impact of climate change on polar bears
Output: The impact of climate change on polar bears is a complex and multifaceted issue that requires a deep understanding of both environmental and anthropological perspectives

Prompt: The benefits of regular exercise
Output: The benefits of regular exercise for the elderly are numerous. For example, physical activity can improve blood circulation, which can reduce the
