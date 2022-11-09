# Simulation modle for longitudinal visual field tests in glaucoma
 This repository is a part of the paper "A Data-driven Model for Simulating Longitudinal Visual Field Tests in Glaucoma"


# Authors
  - Yan Li, Department of Electrical and Computer Engineering, University of Toronto
  - Moshe Eizenman, Department of Ophthalmology & Vision Sciences, University of Toronto
  - Runjie B. Shi, Institute of Biomedical Engineering and Temerty Faculty of Medicine, University of Toronto
  - Yvonne Buys, Department of Ophthalmology & Vision Sciences, University of Toronto
  - Graham Trope, Department of Ophthalmology & Vision Sciences, University of Toronto
  - Willy Wong, Department of Electrical and Computer Engineering and Institute of Biomedical Engineering, University of Toronto


# Usage
Load the **Rotterdam**[[1]](#1) and **Washington**[[2]](#2) longitudinal visual field datasets and process the baseline VF tests:
```
data_list_rt  = load_all_rt_by_eyes(pt_list=None, ceiling=ceiling, min_len=10, cut_type=None, fu_year=5, md_range=None, verbose=1)
data_list_uw  = load_all_uw_by_eyes(pt_list=None, ceiling=ceiling, min_len=10, cut_type=None, fu_year=5, md_range=None, verbose=1)
data_list_all = data_list_rt + data_list_uw
vf_simulator = Longitudinal_VF_simulator()
vf_data_list = vf_simulator.process_baseline(data_list_all)
```

Simulate stable and progressing VF sequences based on the given baseline tests
```
simulated_stable_data, simulated_progress_data = vf_simulator.simulate(vf_data_list, sim_len=15, test_interval=0.5, verbose=0)
```

Visualize simulation VF sequences
```
selected_eye = [218]
vf_simulator.visualize_var_rates(vf_data_list, repeat_per_eye=100, sim_len=15, test_interval=0.5, progress_rate='random', selected_eye=selected_eye) 
```

# License
----
MIT


## References
<a id="1">[1]</a> 
Bryan, S. R. (2013). Robust and censored modeling and prediction of progression in glaucomatous visual fields.  
Investigative ophthalmology & visual science. 2013 Oct 1;54(10):6694-700.

<a id="1">[2]</a> 
Montesano, G. (2022). Open Source Dataset of Perimetry Tests From the Humphrey Field Analyzer at the University of Washington.
Translational vision science & technology. 2022 Jan 3;11(1):2-.
