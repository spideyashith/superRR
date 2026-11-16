# Stage 7 outputs (hand-off to an AI agent)

Stage 7 redoes the B1 bootstrap correctly, fills final_rankings, and adds placebo, 
temporal-holdout, role-split analyses, figures and tests.

**Core conventions**: bowler reported value = minus the fitted coefficient (ridge fit on y_bat). 
Primary spec: B_trim weights, n_eff>=150 and >=15 matches, equal P/C/S weights, alpha 1e4, 
signal variance fixed at Step 6 values (batter 1.4855e-05, bowler 1e-06). 
Rating = mean of z(P), z(C), z(S).


## 1. Basic bootstrap (1000 by-match resamples, seed 42)

- Resamples: 1000; every resample a permutation of 1..n
- Player set: 131 batters, 133 bowlers (fixed = file of record)
- Gates: PASS (1000/1000, full count each resample)
- Sanity: Rashid Khan p_top10=0.989, Sunil Narine (bowler)=0.912, Wanindu Hasaranga=0.793


### Bootstrap top 15 — bowlers

      player_name  median_rank  rank_lo  rank_hi  p_top10  p_top5  p_top10_neff200
      Rashid Khan          1.0      1.0    7.025    0.989   0.938            0.989
     Sunil Narine          4.0      1.0   17.000    0.912   0.652            0.912
    Kuldeep Yadav          6.0      1.0   55.000    0.666   0.438            0.666
Wanindu Hasaranga          6.0      1.0   27.025    0.793   0.495            0.793
       Axar Patel          7.0      1.0   36.000    0.669   0.382            0.669
Bhuvneshwar Kumar          7.0      1.0   70.075    0.589   0.423            0.589
   Josh Hazlewood         11.0      1.0   62.025    0.485   0.268            0.485
       Mark Adair         12.0      2.0   54.050    0.434   0.181            0.434
   Jasprit Bumrah         13.0      2.0   64.025    0.406   0.175            0.406
 Mujeeb Ur Rahman         14.0      2.0   57.025    0.378   0.157            0.378
      Abrar Ahmed         16.0      4.0   48.025    0.260   0.074            0.260
   Shaheen Afridi         18.0      3.0   68.000    0.266   0.093            0.266
Mustafizur Rahman         22.0      3.0   86.000    0.232   0.087            0.232
     Taskin Ahmed         22.0      6.0   63.025    0.143   0.019            0.143
    Sikandar Raza         22.0      6.0   60.000    0.131   0.021            0.131

### Bootstrap top 15 — batters

         player_name  median_rank  rank_lo  rank_hi  p_top10  p_top5  p_top10_neff200
     Abhishek Sharma          3.0      1.0   23.025    0.858   0.643            0.858
    Suryakumar Yadav          4.0      1.0   18.000    0.881   0.612            0.881
           Tim David          4.0      1.0   18.025    0.888   0.657            0.888
         Virat Kohli          6.0      1.0   20.000    0.769   0.435            0.769
           Phil Salt          6.0      1.0   25.000    0.802   0.471            0.802
Vaibhav Sooryavanshi          6.0      1.0   19.025    0.819   0.463            0.819
         Jos Buttler          8.0      1.0   36.025    0.615   0.314            0.615
    Heinrich Klaasen         10.0      1.0   36.025    0.505   0.209            0.505
    Yashasvi Jaiswal         10.0      2.0   41.000    0.517   0.186            0.517
Namboori Tilak Varma         12.0      2.0   49.025    0.419   0.169            0.419
        Shreyas Iyer         13.0      2.0   50.025    0.394   0.145            0.394
       Glenn Maxwell         13.0      1.0   75.025    0.394   0.208            0.394
      Mitchell Marsh         15.0      3.0   61.025    0.281   0.080            0.281
         Rinku Singh         15.0      3.0   57.025    0.260   0.068            0.260
     Nicholas Pooran         16.0      2.0   68.050    0.281   0.095            0.281


## 2. Final rankings (file of record, ranks 1-15 per role)


### Bowlers

 rank       player_name                                                                                               teams competitions    n_eff  matches      P     C     S  rating  boot_median_rank  p_top10  p_top5
    1       Rashid Khan                                                                         Afghanistan; Gujarat Titans         Both 1274.069      117  3.950 1.404 3.647   3.000               1.0    0.989   0.938
    2      Sunil Narine                                                                               Kolkata Knight Riders     IPL-only  754.183       63  3.993 1.427 2.592   2.671               4.0    0.912   0.652
    3 Wanindu Hasaranga Lucknow Super Giants; Rajasthan Royals; Royal Challengers Bengaluru; Sri Lanka; Sunrisers Hyderabad         Both  884.010       81  2.700 1.540 2.055   2.098               6.0    0.793   0.495
    4  Mujeeb Ur Rahman                                                  Afghanistan; Kolkata Knight Riders; Mumbai Indians         Both  341.946       35  1.891 1.309 1.396   1.532              14.0    0.378   0.157
    5       Abrar Ahmed                                                                                            Pakistan    T20I-only  354.046       32  1.915 1.835 1.160   1.637              16.0    0.260   0.074
    6     Piyush Chawla                                                                                      Mumbai Indians     IPL-only  416.528       26  3.698 1.031 0.417   1.715              34.0    0.051   0.010
    7    Shaheen Afridi                                                                                            Pakistan    T20I-only  263.295       42  0.794 0.712 1.415   0.974              18.0    0.266   0.093
    8        Mark Adair                                                                                             Ireland    T20I-only  237.703       33  0.169 0.748 1.928   0.948              12.0    0.434   0.181
    9      Taskin Ahmed                                                                                          Bangladesh    T20I-only  395.281       43  1.078 1.112 1.061   1.084              22.0    0.143   0.019
   10        Axar Patel                                                                               Delhi Capitals; India         Both 1248.868      126  1.039 0.541 2.253   1.278               7.0    0.669   0.382
   11   Lockie Ferguson       Gujarat Titans; Kolkata Knight Riders; New Zealand; Punjab Kings; Royal Challengers Bengaluru         Both  480.506       51  2.583 1.024 0.492   1.367              33.0    0.058   0.012
   12    Josh Hazlewood                                                              Australia; Royal Challengers Bengaluru         Both  520.949       57  0.123 1.546 1.878   1.182              11.0    0.485   0.268
   13     Kuldeep Yadav                                                                               Delhi Capitals; India         Both  865.264       84 -0.088 1.416 2.353   1.227               6.0    0.666   0.438
   14     Krunal Pandya                                                   Lucknow Super Giants; Royal Challengers Bengaluru     IPL-only  649.029       62  2.329 0.133 1.134   1.199              26.0    0.151   0.036
   15 Bhuvneshwar Kumar                                             India; Royal Challengers Bengaluru; Sunrisers Hyderabad         Both  519.062       94 -0.095 0.362 2.696   0.987               7.0    0.589   0.423

### Batters

 rank          player_name                                                                       teams competitions    n_eff  matches     P     C     S  rating  boot_median_rank  p_top10  p_top5
    1 Vaibhav Sooryavanshi                                                     India; Rajasthan Royals         Both  304.467       26 2.552 3.459 2.483   2.831               6.0    0.819   0.463
    2            Tim David                      Australia; Mumbai Indians; Royal Challengers Bengaluru         Both  332.459       91 1.009 1.932 3.498   2.147               4.0    0.888   0.657
    3            Phil Salt Delhi Capitals; England; Kolkata Knight Riders; Royal Challengers Bengaluru         Both  611.908       60 1.979 1.893 2.763   2.212               6.0    0.802   0.471
    4      Abhishek Sharma                                                  India; Sunrisers Hyderabad         Both  911.385       94 1.659 2.278 3.462   2.466               3.0    0.858   0.643
    5          Virat Kohli                                          India; Royal Challengers Bengaluru         Both 1158.234       78 1.466 1.239 2.614   1.773               6.0    0.769   0.435
    6     Suryakumar Yadav                                                       India; Mumbai Indians         Both 1257.730      124 1.393 0.865 3.314   1.858               4.0    0.881   0.612
    7     Heinrich Klaasen                                           South Africa; Sunrisers Hyderabad         Both  506.715       69 1.008 1.112 2.165   1.428              10.0    0.505   0.209
    8     Yashasvi Jaiswal                                                     India; Rajasthan Royals         Both  801.767       57 1.718 1.194 2.004   1.639              10.0    0.517   0.186
    9          Jos Buttler                                   England; Gujarat Titans; Rajasthan Royals         Both  958.968      102 0.847 1.582 2.392   1.607               8.0    0.615   0.314
   10         Shreyas Iyer                                  India; Kolkata Knight Riders; Punjab Kings         Both  746.562       86 1.981 0.207 1.887   1.358              13.0    0.394   0.145
   11 Namboori Tilak Varma                                                       India; Mumbai Indians         Both  883.735      111 1.297 0.441 1.984   1.241              12.0    0.419   0.169
   12          Travis Head                                              Australia; Sunrisers Hyderabad         Both  432.678       48 1.611 1.745 1.213   1.523              17.0    0.197   0.044
   13       Mitchell Marsh                             Australia; Delhi Capitals; Lucknow Super Giants         Both  811.904       70 0.473 1.068 1.785   1.109              15.0    0.281   0.080
   14          Rinku Singh                                                India; Kolkata Knight Riders         Both  239.702       77 1.021 0.525 1.759   1.102              15.0    0.260   0.068
   15        Glenn Maxwell                        Australia; Punjab Kings; Royal Challengers Bengaluru         Both  216.624       64 0.804 0.075 2.175   1.018              13.0    0.394   0.208

## 3. High-pressure stats (pooled LI>=1.5) — top 15


### Batters (player vs league)

 rank          player_name  hp_balls_faced   hp_sr  hp_dot_pct  hp_boundary_pct  league_sr  league_dot  league_bnd
    1 Vaibhav Sooryavanshi           104.0 226.923      33.654           39.423    136.109      39.658       19.08
    2            Tim David           214.0 199.533      27.570           27.570    136.109      39.658       19.08
    3            Phil Salt           383.0 158.747      38.381           26.893    136.109      39.658       19.08
    4      Abhishek Sharma           435.0 159.540      41.609           26.207    136.109      39.658       19.08
    5          Virat Kohli           430.0 153.721      34.186           23.721    136.109      39.658       19.08
    6     Suryakumar Yadav           360.0 156.111      33.611           24.167    136.109      39.658       19.08
    7     Heinrich Klaasen           134.0 200.746      20.149           31.343    136.109      39.658       19.08
    8     Yashasvi Jaiswal           435.0 161.149      38.391           27.356    136.109      39.658       19.08
    9          Jos Buttler           321.0 153.583      37.383           24.611    136.109      39.658       19.08
   10         Shreyas Iyer           263.0 159.696      28.897           20.152    136.109      39.658       19.08
   11 Namboori Tilak Varma           412.0 148.058      32.524           19.660    136.109      39.658       19.08
   12          Travis Head           271.0 147.970      50.185           27.675    136.109      39.658       19.08
   13       Mitchell Marsh           452.0 151.991      42.699           23.230    136.109      39.658       19.08
   14          Rinku Singh           121.0 174.380      24.793           26.446    136.109      39.658       19.08
   15        Glenn Maxwell           122.0 187.705      33.607           31.148    136.109      39.658       19.08

### Bowlers (player vs league)

 rank       player_name  hp_legal_balls  hp_econ  hp_dot_pct  hp_wkts  hp_wkts_per_over  league_econ  league_dot  league_wpo
    1       Rashid Khan           322.0    7.025      37.888     20.0             0.373        8.547      39.839        0.31
    2      Sunil Narine           225.0    6.000      43.111     14.0             0.373        8.547      39.839        0.31
    3 Wanindu Hasaranga           246.0    7.902      35.366     18.0             0.439        8.547      39.839        0.31
    4  Mujeeb Ur Rahman           205.0    7.288      43.902     11.0             0.322        8.547      39.839        0.31
    5       Abrar Ahmed           112.0    6.804      43.750      6.0             0.321        8.547      39.839        0.31
    6     Piyush Chawla            12.0   10.500      25.000      2.0             1.000        8.547      39.839        0.31
    7    Shaheen Afridi           282.0    7.021      46.099     18.0             0.383        8.547      39.839        0.31
    8        Mark Adair           200.0    6.870      45.500     14.0             0.420        8.547      39.839        0.31
    9      Taskin Ahmed           149.0    6.604      51.678      7.0             0.282        8.547      39.839        0.31
   10        Axar Patel           312.0    7.154      40.385     14.0             0.269        8.547      39.839        0.31
   11   Lockie Ferguson           163.0    8.761      48.466     12.0             0.442        8.547      39.839        0.31
   12    Josh Hazlewood           251.0    7.841      48.606     22.0             0.526        8.547      39.839        0.31
   13     Kuldeep Yadav           179.0    7.073      31.844     13.0             0.436        8.547      39.839        0.31
   14     Krunal Pandya           177.0    6.373      36.158     14.0             0.475        8.547      39.839        0.31
   15 Bhuvneshwar Kumar           583.0    7.780      47.856     25.0             0.257        8.547      39.839        0.31


## 4. C1 placebo (100 weight permutations within innings x phase, seed 42)

  role     scheme metric  n_players  real  placebo_mean  placebo_p95  share_placebo_ge_real
batter   odd_even      P        171 0.062         0.129        0.225                   0.87
batter   odd_even      C        171 0.151         0.174        0.278                   0.65
batter early_late      P        126 0.203         0.098        0.223                   0.12
batter early_late      C        126 0.222         0.114        0.237                   0.06
bowler   odd_even      P        194 0.113         0.115        0.227                   0.54
bowler   odd_even      C        194 0.270         0.149        0.244                   0.01
bowler early_late      P        126 0.072         0.110        0.240                   0.66
bowler early_late      C        126 0.158         0.124        0.260                   0.31


## 5. C2 temporal holdout (<2025-01-01 fit, >=2025-01-01 validate; 459 / 307 matches)

  role                 predictor     target  n_players  spearman  pearson                                      note
batter                    rating    raw_apv        130     0.090    0.075                                       NaN
batter                    rating con_median        129     0.120    0.132                                       NaN
batter                         S    raw_apv        130     0.093    0.060                                       NaN
batter                         S con_median        129     0.124    0.135                                       NaN
batter early_top10_mean_late_raw    raw_apv          9     0.001   -0.004  mean_late_raw=0.0011 CI=[-0.0045,0.0060]
batter early_top20_mean_late_raw    raw_apv         19     0.003   -0.004  mean_late_raw=0.0028 CI=[-0.0038,0.0113]
bowler                    rating    raw_apv        127     0.132    0.045                                       NaN
bowler                    rating con_median        127     0.378    0.272                                       NaN
bowler                         S    raw_apv        127     0.134    0.055                                       NaN
bowler                         S con_median        127     0.340    0.261                                       NaN
bowler early_top10_mean_late_raw    raw_apv          7    -0.002   -0.012 mean_late_raw=-0.0016 CI=[-0.0122,0.0076]
bowler early_top20_mean_late_raw    raw_apv         12    -0.002   -0.007 mean_late_raw=-0.0015 CI=[-0.0071,0.0040]


## 6. C3 role splits
### IPL-only / T20I-only rerun top 15


**IPL-only — batters:** Vaibhav Sooryavanshi, Virat Kohli, Shreyas Iyer, Heinrich Klaasen, Nicholas Pooran, Tim David, Jos Buttler, Phil Salt, Shashank Singh, Abhishek Sharma, Shubman Gill, Rinku Singh, Yashasvi Jaiswal, Axar Patel, Travis Head

**IPL-only — bowlers:** Sunil Narine, Bhuvneshwar Kumar, Rashid Khan, Jasprit Bumrah, Kuldeep Yadav, Krunal Pandya, Josh Hazlewood, Trent Boult, Piyush Chawla, Axar Patel, Mohammed Shami, Ravichandran Ashwin, Wanindu Hasaranga, Mohsin Khan, Yuzvendra Chahal

**T20I-only — batters:** Abhishek Sharma, Suryakumar Yadav, Tim David, Phil Salt, Namboori Tilak Varma, Harry Brook, Mitchell Marsh, Dasun Shanaka, Jos Buttler, Daryl Mitchell, Cameron Green, Tim Seifert, Jacob Bethell, Hardik Pandya, Kamindu Mendis

**T20I-only — bowlers:** Rashid Khan, Wanindu Hasaranga, Mark Adair, Axar Patel, Mujeeb Ur Rahman, Sufiyan Muqeem, Mustafizur Rahman, Varun Chakravarthy, Abrar Ahmed, Sikandar Raza, Shaheen Afridi, Lockie Ferguson, Taskin Ahmed, Mohammad Nawaz, Kuldeep Yadav


### Phase share of pressure-weighted balls (top 15)

         player_name  rank  death_share  middle_share  powerplay_share
     Abhishek Sharma     4        0.019         0.338            0.643
       Glenn Maxwell    15        0.277         0.609            0.114
    Heinrich Klaasen     7        0.311         0.615            0.074
         Jos Buttler     9        0.147         0.420            0.433
      Mitchell Marsh    13        0.079         0.444            0.477
Namboori Tilak Varma    11        0.320         0.527            0.153
           Phil Salt     3        0.057         0.370            0.573
         Rinku Singh    14        0.603         0.353            0.044
        Shreyas Iyer    10        0.197         0.592            0.212
    Suryakumar Yadav     6        0.182         0.605            0.214
           Tim David     2        0.651         0.312            0.037
         Travis Head    12        0.000         0.266            0.734
Vaibhav Sooryavanshi     1        0.019         0.346            0.634
         Virat Kohli     5        0.121         0.490            0.389
    Yashasvi Jaiswal     8        0.019         0.450            0.531
         Abrar Ahmed     5        0.132         0.641            0.228
          Axar Patel    10        0.095         0.666            0.239
   Bhuvneshwar Kumar    15        0.527         0.094            0.378
      Josh Hazlewood    12        0.390         0.178            0.432
       Krunal Pandya    14        0.061         0.800            0.139
       Kuldeep Yadav    13        0.183         0.809            0.008
     Lockie Ferguson    11        0.275         0.400            0.326
          Mark Adair     8        0.564         0.071            0.365
    Mujeeb Ur Rahman     4        0.174         0.369            0.457
       Piyush Chawla     6        0.048         0.791            0.161
         Rashid Khan     1        0.226         0.658            0.116
      Shaheen Afridi     7        0.458         0.153            0.389
        Sunil Narine     2        0.148         0.682            0.169
        Taskin Ahmed     9        0.389         0.295            0.316
   Wanindu Hasaranga     3        0.192         0.756            0.052


### Batter position-bucket share of pressure weight

        striker_name   1-2   3-5    6+
     Abhishek Sharma 0.957 0.043 0.000
       Glenn Maxwell 0.027 0.749 0.223
    Heinrich Klaasen 0.000 0.805 0.195
         Jos Buttler 0.578 0.418 0.003
      Mitchell Marsh 0.554 0.446 0.000
Namboori Tilak Varma 0.000 0.923 0.077
           Phil Salt 0.963 0.017 0.020
         Rinku Singh 0.000 0.446 0.554
        Shreyas Iyer 0.015 0.979 0.006
    Suryakumar Yadav 0.027 0.973 0.000
           Tim David 0.000 0.527 0.473
         Travis Head 1.000 0.000 0.000
Vaibhav Sooryavanshi 0.965 0.035 0.000
         Virat Kohli 0.793 0.207 0.000
    Yashasvi Jaiswal 1.000 0.000 0.000


## 7. E tests and protected hashes

- `tests/test_step7.py`: 6 tests pass (resample count/differ, permutation, weight-multiset, 
  temporal split disjoint, final_rankings uniqueness/ranks/top10/registry, half-frame signs, protected hashes).
- Full suite: **91 passed**.
- Protected-hash check: 14 files; 7 with a pre-step4 baseline all match; 7 later outputs report current hash only.
- **Known data issue**: frozen `step6_half_components.csv` is truncated (only batter-B and bowler-B_trim). 
  Use `step6_half_components_rebuilt2.csv` (all 4 role x variant combos); same-half rho positive: batters 0.66-0.75, bowlers 0.84-0.89.


## 8. Honest limitations

- C1: real pressure weighting beats placebo only on the early/late split; on odd/even it does not.
- C2: predictive correlations are weak (rho ~0.09-0.38) and early top-10/top-20 CIs include zero.
- All four T20I-only bowlers (Abrar Ahmed, Shaheen Afridi, Mark Adair, Taskin Ahmed) remain in the T20I-only top 15.
- Sunil Narine's batter p_top10 is 0.0 by design; he is a bowler (p_top10=0.912).


## 9. Output files

- `outputs/final_rankings.csv`
- `outputs/unstop_entry.txt`
- `outputs/tables/s7_bootstrap_summary.csv`
- `outputs/tables/s7_bootstrap_ranks.csv`
- `outputs/tables/s7_c1_placebo.csv`
- `outputs/tables/s7_c1_distributions.csv`
- `outputs/tables/s7_c2_temporal.csv`
- `outputs/tables/s7_c3_phase_share.csv`
- `outputs/tables/s7_c3_batter_buckets.csv`
- `outputs/tables/s7_c3_reruns.csv`
- `outputs/tables/step6_half_components_rebuilt2.csv`
- `outputs/figures/s7_forest_rank_intervals.png`
- `outputs/figures/s7_S_vs_P_scatter.png`
- `outputs/figures/s7_bump_variants.png`
- `outputs/figures/s7_real_vs_placebo_hist.png`