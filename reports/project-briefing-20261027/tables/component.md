| arm | prefill批次 | query tokens | prefill attention累计 ms | writer累计 ms | writer占kernel累计 % | Graph decode批次 | Graph整模型/批中位 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bf16 | 17 | 138589 | 2090.391 | 51.029 | 0.614 | 31 | 20.063 |
| fp8_e4m3 | 18 | 149546 | 4016.832 | 40.491 | 0.399 | 30 | 20.212 |
