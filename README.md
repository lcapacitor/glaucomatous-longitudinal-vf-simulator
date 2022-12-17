# Simulation model for longitudinal visual field tests in glaucoma
 This repository is a part of the paper "A Data-driven Model for Simulating Longitudinal Visual Field Tests in Glaucoma"

![vf_simulator](https://github.com/lcapacitor/glaucomatous-longitudinal-vf-simulator/blob/main/figures/sim_eye_218/eye_218_1090.png)

Panel (a): Visual field (VF) sequence of a glaucoma patient over a period of 7 years. Panels (b)-(e): Simulated VF sequences for the same baseline VF test with different progression rates. The patient data show moderate VF progression loss with MD linear regression slope of −0.5 dB/year. Panels (b) shows a simulated stable VF sequence (MD slope of 0 dB/year). Panel (c) shows a simulated VF sequence with the same progression rate (MD slope of −0.5 dB/year) as the patient’s data. Panels (d) and (e) show simulated VF sequences with higher progression rates than that of the patient’s data (MD slopes of −1.0 dB/year and −1.5 dB/year, respectively). 


## Authors
  - **Yan Li**, Department of Electrical & Computer Engineering, University of Toronto
  - **Moshe Eizenman**, Department of Ophthalmology & Vision Sciences, University of Toronto
  - **Runjie B. Shi**, Institute of Biomedical Engineering and Temerty Faculty of Medicine, University of Toronto
  - **Yvonne Buys**, Department of Ophthalmology & Vision Sciences, University of Toronto
  - **Graham Trope**, Department of Ophthalmology & Vision Sciences, University of Toronto
  - **Willy Wong**, Department of Electrical and Computer Engineering and Institute of Biomedical Engineering, University of Toronto


## Usage
#### Load the **Rotterdam**[[1]](#1) longitudinal visual field datasets and process the baseline VF tests:
```
data_list_rt = load_all_rt_by_eyes(pt_list=None, ceiling=ceiling, min_len=10, cut_type=None, fu_year=5, md_range=None, verbose=1)
vf_data_list = vf_simulator.process_baseline(data_list_rt)
```

#### Simulate stable and progressing VF sequences based on the given baseline tests
```
simulated_stable_data, simulated_progress_data = vf_simulator.simulate(vf_data_list, sim_len=15, test_interval=0.5, verbose=0)
```
When setting ```verbose=1```, the textual simulation log is provided (the selection of progression centers/clusters).
When setting ```verbose=2```, each simulated VF sequence is visualized, including the simulated progression patterns. 

#### Example of a simulated progressing VF sequence:
![prog_eye](https://github.com/lcapacitor/glaucomatous-longitudinal-vf-simulator/blob/main/figures/sim_prog/prog_eye_01.png)


#### Example of a simulated stable VF sequence:
![stable_eye](https://github.com/lcapacitor/glaucomatous-longitudinal-vf-simulator/blob/main/figures/sim_stable/stable_eye_01.png)



#### Visualize simulation VF sequences
```
vf_simulator.visualize_var_rates(vf_data_list, repeat_per_eye=1000, selected_eye=[218], progress_rate=[0,-1.5,-1.6,-2.5], progress_cluster=['stable',3,6,8])
```



## License
MIT


## References
<a id="1">[1]</a> 
Bryan SR, Vermeer KA, Eilers PH, Lemij HG, Lesaffre EM. Robust and censored modeling and prediction of progression in glaucomatous visual fields. Investigative ophthalmology & visual science. 2013 Oct 1;54(10):6694-700.
