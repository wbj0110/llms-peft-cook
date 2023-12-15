option = {
  title: {
    text: ''
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
      data: [97.6, 98.2, 98.7, 99.6, 99.6]
    },
    {
      name: 'BERT_large(LoRA)',
      type: 'line',
      stack: 'Total',
      data: [76.8, 79.2, 96.7, 97.2, 98.2]
    },
    {
      name: 'FLAN-T5_base(LoRA)',
      type: 'line',
      stack: 'Total',
      data: [63.6, 87.2, 89.4, 90.4, 91.7]
    },
    {
      name: 'FLAN-T5_large(LoRA)',
      type: 'line',
      stack: 'Total',
      data: [81.4, 93.6, 97.0, 97.1, 97.4]
    },
    {
      name: 'FLAN-T5_xl(LoRA)',
      type: 'line',
      stack: 'Total',
      data: [91.2, 93.6, 97.3, 97.6, 98.2]
    }
  ]
};