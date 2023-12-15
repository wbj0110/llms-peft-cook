option = {
  title: {
    text: 'Stacked Line'
  },
  tooltip: {
    trigger: 'axis'
  },
  legend: {
    data: ['BERT_base(LoRA)', 'BERT_large(LoRA)', 'FLAN-T5_base(LoRA)', 'FLAN-T5_large(LoRA)', 'FLAN-T5_xl(LoRA)']
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true
  },
  toolbox: {
    feature: {
      saveAsImage: {}
    }
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: ['Epoch1', 'Epoch2', 'Epoch3', 'Epoch4', 'Epoch5']
  },
  yAxis: {
    type: 'value'
  },
  series: [
    {
      name: 'BERT_base(LoRA)',
      type: 'line',
      stack: 'Total',
      data: [87.3, 89.2, 90.2, 91.1, 91.7]
    },
    {
      name: 'BERT_large(LoRA)',
      type: 'line',
      stack: 'Total',
      data: [88.3, 90.4, 92.1, 93.5, 93.6]
    },
    {
      name: 'FLAN-T5_base(LoRA)',
      type: 'line',
      stack: 'Total',
      data: [93.5, 94.4, 94.6, 95.0, 95.1]
    },
    {
      name: 'FLAN-T5_large(LoRA)',
      type: 'line',
      stack: 'Total',
      data: [93.6, 94.6, 94.8, 95.1, 95.1]
    },
    {
      name: 'FLAN-T5_xl(LoRA)',
      type: 'line',
      stack: 'Total',
      data: [94.2, 95.4, 95.9, 96.4, 96.8]
    }
  ]
};