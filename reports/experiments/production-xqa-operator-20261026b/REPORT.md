# XQA第二次验收：测试夹具页表别名错误

confirmed：acbf3478在H800/GPU0的padded Graph参考处5012/10240元素不符、maxabs28.197。它已执行此前的finite初始化、prefill和B2 Graph/replan检查，但没有完整operator通过seal，未加载14B、未启动BF16、无质量或性能结果。GPU释放及额度结算已确认，50份失败证据逐SHA镜像。

CPU复现确定本次错误来自我新增的测试夹具。benchmark helper的pages.reshape(-1).to(int32)在原pages已经int32时共享storage；测试为重规划写入indices时，把原[[2,0,4],[3,1,5]]变为[[2,0,3],[1,1,5]]。随后构造B4页表时得到[[2,0],[1,1]]，第二真实序列把已填极大残值的末页当作完整第一页，解释约28的错误输出。

因此此错误不能作为XQA padded kernel缺陷的证据。d60812fd只为planner建立独立indices storage，保持reference页表不可变，并加入对应CPU回归。93项源码CPU通过、1项CUDA跳过；GPU复验仍待新c。原a的NaN尾页现象单独保留，有限初始化及同Graph NaN K/V对照仍需实际完成。

原b失败终态/源码/日志/编译cache保持原样，不重启或覆盖。修正夹具不降低数值门槛、不改变真实待测引擎配置；下一次必须新run。
