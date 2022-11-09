# Simulation modle for longitudinal visual field tests in glaucoma
 This repository is a part of the paper "A Data-driven Model for Simulating Longitudinal Visual Field Tests in Glaucoma"

# Usage
Load the **Rotterdam**[[1]](#1) and **Washington**[[2]](#2) longitudinal visual field datasets and process the baseline VF tests:
```
data_list_rt  = load_all_rt_by_eyes(pt_list=None, ceiling=ceiling, min_len=10, cut_type=None, fu_year=5, md_range=None, verbose=1)
data_list_uw  = load_all_uw_by_eyes(pt_list=None, ceiling=ceiling, min_len=10, cut_type=None, fu_year=5, md_range=None, verbose=1)
data_list_all = data_list_rt + data_list_uw
vf_simulator = Longitudinal_VF_simulator()
vf_data_list = vf_simulator.process_baseline(data_list_all)
```


## References
<a id="1">[1]</a> 
Bryan, S. R. (2013). Robust and censored modeling and prediction of progression in glaucomatous visual fields.  
Investigative ophthalmology & visual science. 2013 Oct 1;54(10):6694-700.
<a id="1">[2]</a> 
Montesano, G. (2022). Open Source Dataset of Perimetry Tests From the Humphrey Field Analyzer at the University of Washington.
Translational vision science & technology. 2022 Jan 3;11(1):2-.
