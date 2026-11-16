# Stage 4: Final Player Performance Report

Generated from saved CSV/Parquet outputs by `py -m src.stage4_report`.
No number in this report is hand-entered.

## 1. Innings-1 model selection and freeze

Two innings-1 candidates were scored on identical out-of-fold deliveries and the
same GroupKFold-by-match assignment.

```
          model  n_deliveries  n_matches  n_folds  log_loss    brier      auc      ece  weighted_abs_calibration_error
logistic_spline         95425        766        5  0.635424 0.222548 0.690111 0.019547                        0.019547
       lightgbm         95425        766        5  0.642101 0.224952 0.681637 0.019737                        0.019737
```


Calibration by predicted-WP decile:

```
 decile    n     pred      obs    signed      abs           model  n_matches
      1 9543 0.198559 0.177408 -0.021152 0.021152 logistic_spline        240
      2 9542 0.312689 0.310627 -0.002063 0.002063 logistic_spline        401
      3 9543 0.382172 0.399874  0.017702 0.017702 logistic_spline        531
      4 9542 0.434102 0.450744  0.016642 0.016642 logistic_spline        634
      5 9543 0.475788 0.457927 -0.017861 0.017861 logistic_spline        641
      6 9542 0.521123 0.535842  0.014719 0.014719 logistic_spline        712
      7 9542 0.562742 0.581010  0.018268 0.018268 logistic_spline        656
      8 9543 0.610822 0.641936  0.031114 0.031114 logistic_spline        557
      9 9542 0.680297 0.676273 -0.004024 0.004024 logistic_spline        413
     10 9543 0.800010 0.748088 -0.051922 0.051922 logistic_spline        269
      1 9543 0.168470 0.184638  0.016168 0.016168        lightgbm        252
      2 9542 0.295485 0.311151  0.015666 0.015666        lightgbm        393
      3 9543 0.389123 0.424814  0.035691 0.035691        lightgbm        507
      4 9542 0.455571 0.434710 -0.020861 0.020861        lightgbm        641
      5 9543 0.496271 0.493241 -0.003030 0.003030        lightgbm        714
      6 9542 0.530523 0.512890 -0.017632 0.017632        lightgbm        701
      7 9542 0.569079 0.573674  0.004595 0.004595        lightgbm        628
      8 9543 0.616274 0.644661  0.028387 0.028387        lightgbm        532
      9 9542 0.673141 0.656257 -0.016884 0.016884        lightgbm        426
     10 9543 0.782141 0.743686 -0.038455 0.038455        lightgbm        273
```


**Decision:** logistic-spline innings 1 (better OOF log loss, Brier, AUC and ECE)
and monotone LightGBM (Model D) innings 2. The two production artifacts are
`wp_fold{k}_innings1_logistic_spline.joblib` and
`wp_fold{k}_innings2_lightgbm_mono.joblib`; final models are
`wp_innings1_logistic_spline.joblib` and `wp_innings2_lightgbm_mono.joblib`.
The freeze manifest records file hashes.

## 2. Frozen models and definitions

- Innings 1: logistic regression + cubic splines, 5-fold GroupKFold by match.
- Innings 2: Model D, LightGBM with monotone constraints, same folds.
- Variant A: within-innings continuous LI (`li_innings`).
- Variant B (main): pooled continuous LI (`li_pooled`).
- Variant C: innings-2-only continuous LI (`li_innings`, innings 1 weight 0).
- Ordinary: `ordinary_weight = 1` for pooled LI at or below the pooled median.

LI by over:

```
          model  innings  over    n  mean_li_raw  mean_li_pooled  median_li   p75_li   p90_li
logistic_spline        1     1 4849     0.022112        1.442124   0.019789 0.025363 0.037959
logistic_spline        1     2 4827     0.011512        0.750810   0.011338 0.013944 0.015955
logistic_spline        1     3 4765     0.010928        0.712720   0.011320 0.013109 0.014312
logistic_spline        1     4 4763     0.011342        0.739715   0.011892 0.013303 0.014676
logistic_spline        1     5 4756     0.011738        0.765532   0.012374 0.013562 0.014830
logistic_spline        1     6 4747     0.011768        0.767504   0.012336 0.013484 0.014710
logistic_spline        1     7 4714     0.009448        0.616198   0.009884 0.010715 0.011672
logistic_spline        1     8 4734     0.009130        0.595439   0.009484 0.010316 0.011345
logistic_spline        1     9 4737     0.008796        0.573644   0.009127 0.010035 0.010877
logistic_spline        1    10 4757     0.008450        0.551115   0.008739 0.009793 0.010437
logistic_spline        1    11 4774     0.009646        0.629131   0.009719 0.011347 0.012272
logistic_spline        1    12 4776     0.009208        0.600523   0.009251 0.010766 0.012044
logistic_spline        1    13 4790     0.008817        0.575040   0.008955 0.010123 0.011579
logistic_spline        1    14 4769     0.008428        0.549655   0.008639 0.009571 0.010797
logistic_spline        1    15 4741     0.008045        0.524692   0.008286 0.009337 0.010139
logistic_spline        1    16 4798     0.009418        0.614245   0.009849 0.011144 0.011753
logistic_spline        1    17 4837     0.009190        0.599349   0.009724 0.010847 0.011551
logistic_spline        1    18 4818     0.008908        0.581006   0.009482 0.010682 0.011427
logistic_spline        1    19 4827     0.013008        0.848380   0.013711 0.016444 0.018277
logistic_spline        1    20 4646     0.013366        0.871695   0.013907 0.017599 0.019837
        model_D        2     1 4802     0.043135        2.813219   0.038874 0.059282 0.075696
        model_D        2     2 4833     0.026422        1.723252   0.022582 0.036399 0.050190
        model_D        2     3 4774     0.022005        1.435146   0.019711 0.030722 0.041431
        model_D        2     4 4784     0.019617        1.279428   0.017403 0.027355 0.037971
        model_D        2     5 4780     0.018862        1.230150   0.017084 0.026276 0.036519
        model_D        2     6 4756     0.017857        1.164643   0.015902 0.025131 0.035497
        model_D        2     7 4705     0.015738        1.026447   0.013871 0.022251 0.031124
        model_D        2     8 4735     0.014987        0.977414   0.013144 0.021583 0.030369
        model_D        2     9 4715     0.015055        0.981912   0.013197 0.022272 0.031136
        model_D        2    10 4692     0.015321        0.999243   0.012477 0.023338 0.033382
        model_D        2    11 4639     0.015423        1.005903   0.011869 0.023172 0.035042
        model_D        2    12 4627     0.015516        1.011950   0.010230 0.023267 0.037427
        model_D        2    13 4584     0.015902        1.037131   0.009744 0.024867 0.040078
        model_D        2    14 4529     0.016050        1.046774   0.009049 0.025519 0.041401
        model_D        2    15 4426     0.016755        1.092756   0.008723 0.026667 0.044653
        model_D        2    16 4297     0.020309        1.324517   0.007700 0.034350 0.057101
        model_D        2    17 4044     0.021652        1.412111   0.007370 0.034763 0.065388
        model_D        2    18 3715     0.021419        1.396936   0.005717 0.029917 0.069551
        model_D        2    19 3160     0.026580        1.733567   0.005023 0.036142 0.092831
        model_D        2    20 2145     0.042917        2.799031   0.005896 0.038244 0.164549
       lightgbm        1     1 4849     0.019902        1.202308   0.016355 0.026443 0.038595
       lightgbm        1     2 4827     0.016372        0.989046   0.012945 0.020919 0.031603
       lightgbm        1     3 4765     0.017010        1.027590   0.014040 0.022560 0.031542
       lightgbm        1     4 4763     0.016694        1.008520   0.014024 0.022125 0.030238
       lightgbm        1     5 4756     0.015341        0.926763   0.012870 0.020417 0.027832
       lightgbm        1     6 4747     0.014209        0.858393   0.011703 0.018987 0.027046
       lightgbm        1     7 4714     0.011919        0.720053   0.009643 0.015197 0.022849
       lightgbm        1     8 4734     0.011414        0.689546   0.009637 0.014731 0.021558
       lightgbm        1     9 4737     0.011556        0.698112   0.009882 0.015052 0.022626
       lightgbm        1    10 4757     0.011085        0.669688   0.009320 0.014644 0.021904
       lightgbm        1    11 4774     0.011658        0.704271   0.009607 0.015714 0.023196
       lightgbm        1    12 4776     0.010967        0.662550   0.008589 0.015221 0.022223
       lightgbm        1    13 4790     0.011147        0.673383   0.009270 0.015181 0.021848
       lightgbm        1    14 4769     0.010542        0.636837   0.008620 0.014426 0.021066
       lightgbm        1    15 4741     0.010009        0.604651   0.008148 0.013567 0.020462
       lightgbm        1    16 4798     0.011148        0.673467   0.008874 0.015385 0.023428
       lightgbm        1    17 4837     0.011098        0.670466   0.009192 0.015383 0.022381
       lightgbm        1    18 4818     0.011376        0.687229   0.008685 0.015760 0.025100
       lightgbm        1    19 4827     0.013745        0.830362   0.011201 0.019493 0.028675
       lightgbm        1    20 4646     0.012931        0.781163   0.010148 0.018024 0.027752
        model_D        2     1 4802     0.043135        2.605815   0.038874 0.059282 0.075696
        model_D        2     2 4833     0.026422        1.596206   0.022582 0.036399 0.050190
        model_D        2     3 4774     0.022005        1.329341   0.019711 0.030722 0.041431
        model_D        2     4 4784     0.019617        1.185102   0.017403 0.027355 0.037971
        model_D        2     5 4780     0.018862        1.139457   0.017084 0.026276 0.036519
        model_D        2     6 4756     0.017857        1.078780   0.015902 0.025131 0.035497
        model_D        2     7 4705     0.015738        0.950772   0.013871 0.022251 0.031124
        model_D        2     8 4735     0.014987        0.905355   0.013144 0.021583 0.030369
        model_D        2     9 4715     0.015055        0.909521   0.013197 0.022272 0.031136
        model_D        2    10 4692     0.015321        0.925574   0.012477 0.023338 0.033382
        model_D        2    11 4639     0.015423        0.931743   0.011869 0.023172 0.035042
        model_D        2    12 4627     0.015516        0.937344   0.010230 0.023267 0.037427
        model_D        2    13 4584     0.015902        0.960669   0.009744 0.024867 0.040078
        model_D        2    14 4529     0.016050        0.969601   0.009049 0.025519 0.041401
        model_D        2    15 4426     0.016755        1.012193   0.008723 0.026667 0.044653
        model_D        2    16 4297     0.020309        1.226867   0.007700 0.034350 0.057101
        model_D        2    17 4044     0.021652        1.308004   0.007370 0.034763 0.065388
        model_D        2    18 3715     0.021419        1.293947   0.005717 0.029917 0.069551
        model_D        2    19 3160     0.026580        1.605760   0.005023 0.036142 0.092831
        model_D        2    20 2145     0.042917        2.592673   0.005896 0.038244 0.164549
```


Variant B HIGH shares by phase:

```
 innings1_model  innings     phase     n  high    share  threshold_pooled_q75
logistic_spline        1 powerplay 28707 15581 0.542760              0.787291
logistic_spline        1    middle 42792  1893 0.044237              0.787291
logistic_spline        1     death 23926  6383 0.266781              0.787291
logistic_spline        2 powerplay 28729 20624 0.717881              0.787291
logistic_spline        2    middle 41652 20472 0.491501              0.787291
logistic_spline        2     death 17361  7097 0.408790              0.787291
       lightgbm        1 powerplay 28707 10469 0.364685              1.046094
       lightgbm        1    middle 42792  7758 0.181296              1.046094
       lightgbm        1     death 23926  5630 0.235309              1.046094
       lightgbm        2 powerplay 28729 16659 0.579867              1.046094
       lightgbm        2    middle 41652 15282 0.366897              1.046094
       lightgbm        2     death 17361  6229 0.358793              1.046094
```


## 3. Ball-value methodology

`outputs/tables/ball_values.parquet` has 183,167 deliveries and
`outputs/tables/counterfactual_wp.parquet` stores the post-ball WP for every
outcome. WPAE = actual WPA minus expected WPA:

exp_wpa = sum_k p_out_k * (WP_after,k - WP_pre)

y_bat = WPAE, y_bowler = -y_bat. Secondary metric: batter RAE = runs - exp_runs;
bowler RAE = expected bowler-charged runs - actual bowler-charged runs.

Validation:

```
                       check        value
                        rows 1.831670e+05
              duplicate_keys 0.000000e+00
                  null_y_bat 0.000000e+00
max_abs_li_raw_rebuild_delta 1.110223e-16
              exp_wpa_finite 1.831670e+05
       p_out_sum_max_abs_dev 3.330669e-16
 max_abs_y_bowler_plus_y_bat 0.000000e+00
   ordinary_median_threshold 6.868464e-01
         ordinary_weight_sum 9.158400e+04
       weight_C_innings1_max 0.000000e+00
```


## 4. Ridge model

y = intercept + batter effect + bowler effect + phase + innings + competition + error,
estimated by `sklearn.linear_model.Ridge` with sample weights. Alpha is selected by
5-fold GroupKFold by match; the encoder is fitted on training matches only. Unseen
held-out players receive no player effect. Coefficients are centred so zero is the
average player effect. The intercept is not penalised. No Stage 5 shrinkage.

```
   model   alpha  cv_weighted_mse  baseline_no_player_cv_mse   improvement  n_players_batters  n_players_bowlers  n_train_matches  n_valid_matches
       B 10000.0         0.005561                   0.005560 -1.193780e-06                815                586              612              153
       A 10000.0         0.004309                   0.004309 -6.240130e-07                815                586              612              153
       C 10000.0         0.008413                   0.008411 -2.407255e-06                815                586              612              153
ordinary  1000.0         0.000190                   0.000190  9.940141e-08                815                586              612              153
```


## 5. APV, RR and CON

- Batter APV = centred fitted batter coefficient on WPAE; bowler APV = negative of
  the centred bowler coefficient.
- APV per 100 pressure-weighted balls = APV x 100 (a scaling, not a ratio).
- RR = APV_pressure - APV_ordinary (a difference).
- Bootstrap: 300 match-level resamples, alpha fixed, pressure and ordinary models
  refitted, top 40 eligible players by B pressure APV only (documented shortcut).

Bootstrap summary:

```
player_type  n_requested  n_success_min  n_failed_max
     batter          300            300             0
     bowler          300            300             0
```


CON components (reported separately, not combined):

- Positive share: share of qualifying pressure innings (batters) or per-match
  spells (bowlers) with positive weighted WPAE; a unit needs >= 6 pressure-weighted balls.
- Median per-match pressure value = median of sum(weight x WPAE) over qualifying matches.
- MAD = median absolute deviation of that per-match value.

## 6. Effective sample size

n_eff = (sum w)^2 / sum(w^2) using each variant's LI weight. Also reported: raw
pressure deliveries, deliveries with pooled LI >= 1.5, total weight, matches and
qualifying units.

## 7. Eligibility

```
player_type variant  n_eff_threshold  min_matches  eligible
     batter       B              100           15       179
     batter       B              150           15       161
     batter       B              200           15       130
     batter       A              100           15       190
     batter       A              150           15       175
     batter       A              200           15       155
     batter       C              100           15       110
     batter       C              150           15        85
     batter       C              200           15        60
     bowler       B              100           15       176
     bowler       B              150           15       165
     bowler       B              200           15       134
     bowler       A              100           15       180
     bowler       A              150           15       175
     bowler       A              200           15       164
     bowler       C              100           15        93
     bowler       C              150           15        82
     bowler       C              200           15        56
```


Recommended threshold: **n_eff >= 150 and >= 15 matches**. It retains 161 batters
and 165 bowlers under B while requiring meaningful exposure; 100 admits very
thin samples and 200 discards too many C-variant bowlers.

## 8. Variant ranking correlations

```
player_type variant_1 variant_2  n_common_players  spearman
     batter         B         A                85  0.989173
     batter         B         C                85  0.984581
     batter         A         C                85  0.954700
     bowler         B         A                82  0.995081
     bowler         B         C                82  0.991533
     bowler         A         C                82  0.978059
```


## 9. Top 15 batters under Variant B

```
         player_name  player_id       n_eff  pressure_balls  matches  balls_li_ge_1_5      apv  apv_per_100  apv_ordinary       rr    ci_low  ci_high  con_positive_share  con_median  con_mad
         Rinku Singh       4708  239.702114            1195       77              131 0.003760     0.375957      0.000225 0.003534 -0.000015 0.007637            0.600000    0.022354 0.049961
       Glenn Maxwell        838  232.466667             932       67              126 0.003720     0.371994      0.000045 0.003675 -0.000041 0.009090            0.571429    0.013994 0.083169
           Tim David       6396  332.458580            1211       91              224 0.003696     0.369562      0.001065 0.002631 -0.000778 0.007124            0.720000    0.078490 0.111734
      Shashank Singh       2756  154.032936             583       41               86 0.002240     0.224049      0.000515 0.001725 -0.000938 0.006016            0.666667    0.062195 0.086239
         Jos Buttler        813 1197.428674            2771      115              332 0.002198     0.219774      0.000737 0.001461 -0.001129 0.005569            0.639175    0.026241 0.093854
        Nitish Reddy       8221  151.993146             681       44               62 0.001398     0.139835      0.000180 0.001218 -0.000812 0.004306            0.615385    0.012132 0.051201
         Harry Brook       6280  309.380371             888       52              115 0.001282     0.128211      0.000518 0.000764 -0.001136 0.004028            0.483871   -0.002628 0.127627
       Dasun Shanaka       2436  251.520506             720       53              147 0.001276     0.127599     -0.000104 0.001380 -0.000640 0.003973            0.406250   -0.017064 0.069653
        Shreyas Iyer       3647  778.761750            1802       87              276 0.001161     0.116114      0.000591 0.000570 -0.000631 0.001803            0.517241    0.015604 0.050502
    Suryakumar Yadav       1699 1301.644141            2683      126              374 0.001091     0.109079      0.000982 0.000108 -0.001335 0.001674            0.620690    0.031696 0.090132
Vaibhav Sooryavanshi      50284  302.119524             597       30              110 0.000984     0.098385      0.000728 0.000256 -0.000776 0.001590            0.807692    0.239237 0.197028
Namboori Tilak Varma       7897  901.650544            2345      112              422 0.000957     0.095740      0.000582 0.000375 -0.001317 0.002613            0.586667    0.022979 0.062144
   Washington Sundar       4823  194.487088             633       46              131 0.000934     0.093414      0.000107 0.000827 -0.000577 0.003112            0.500000    0.004957 0.061498
         Virat Kohli         47 1322.010128            2533       88              450 0.000894     0.089429      0.000220 0.000674 -0.000271 0.001613            0.720000    0.075214 0.110906
       Rahul Tewatia       3926  156.527896             553       56              130 0.000891     0.089097     -0.000416 0.001307 -0.001052 0.005007            0.357143   -0.027778 0.126893
```


## 10. Top 15 bowlers under Variant B

```
       player_name  player_id       n_eff  pressure_balls  matches  balls_li_ge_1_5      apv  apv_per_100  apv_ordinary       rr    ci_low  ci_high  con_positive_share  con_median  con_mad
      Tanzim Sakib       7905  167.592066             669       29              118 0.002281     0.228051      0.000361 0.001919 -0.001088 0.006855            0.615385    0.027540 0.073219
     Anrich Nortje       7270  205.515102             841       37               77 0.001922     0.192244      0.000207 0.001715 -0.000576 0.005382            0.305556   -0.066617 0.072173
    Arshdeep Singh       6464 1232.328560            3203      137              737 0.001873     0.187284     -0.000273 0.002146 -0.000467 0.005545            0.527132    0.013767 0.139099
     Andre Russell       1318  291.540099             753       55              152 0.001871     0.187141      0.000006 0.001866 -0.000026 0.004549            0.400000   -0.026269 0.165321
Azmatullah Omarzai       6676  420.876066            1061       62              276 0.001827     0.182741      0.000039 0.001788 -0.000832 0.006111            0.452830   -0.010459 0.132988
    Marcus Stoinis       2639  188.591461             751       52              138 0.001595     0.159517      0.000292 0.001303 -0.001571 0.005661            0.538462    0.022188 0.081033
      Chris Jordan        786  156.252753             574       27               28 0.001457     0.145701      0.000080 0.001377 -0.000597 0.004178            0.300000   -0.039066 0.065139
      Marco Jansen       7808  563.741969            1313       55              155 0.001348     0.134798      0.000246 0.001102 -0.001148 0.004323            0.442308   -0.017018 0.066552
     Vaibhav Arora       8746  307.034792             863       39              218 0.001345     0.134478     -0.000267 0.001612 -0.000384 0.004016            0.315789   -0.057185 0.141704
       Jacob Duffy      22920  237.731160             869       38              184 0.001325     0.132478      0.000090 0.001234 -0.001640 0.005616            0.666667    0.084465 0.194830
     Shahbaz Ahmed       7472  219.770634             533       41               95 0.001222     0.122157      0.000929 0.000292 -0.001007 0.001950            0.230769   -0.034422 0.061259
   Prasidh Krishna       4489  321.861902            1185       50              218 0.001108     0.110843     -0.000054 0.001162 -0.003055 0.006315            0.456522   -0.037718 0.126529
    Jaydev Unadkat       1071  208.197289             551       28              106 0.001073     0.107310     -0.000403 0.001476 -0.000012 0.004010            0.440000   -0.036253 0.081503
      Prince Yadav      58246  267.408390             671       29               71 0.001052     0.105191     -0.000017 0.001069 -0.000382 0.003076            0.357143   -0.048048 0.108276
       Odean Smith       4840  166.230989             387       22               67 0.000850     0.085034      0.000266 0.000585 -0.001259 0.002750            0.470588   -0.017793 0.085019
```


Sensitivity tables for A and C are saved as
`player_metrics_batters_A.csv`, `player_metrics_bowlers_A.csv`,
`player_metrics_batters_C.csv`, `player_metrics_bowlers_C.csv`. They are
sensitivity results, not final rankings.

## 11. Limitations and unresolved items

- Per-ball WPAE is extremely noisy. Ridge CV error is essentially equal to the
  no-player-effects baseline (see diagnostics); player effects are therefore
  weak and APV differences should be read as provisional.
- The bootstrap covers only the top 40 eligible players per role by B APV.
- The extras outcome is a single state transition (+1 run, no legal ball), matching
  the frozen LI calculation; a finer extras subtype model is not implemented.
- Old `player_metrics_aps.csv` is from a different delivery universe and is not
  used; no adjusted-vs-old APV correlation is reported.
- No empirical-Bayes shrinkage (Stage 5) and no PPR ranking (Stage 6).

## 12. Reproducibility

```
py -m src.innings1_compare
py -m src.revert_innings1       # only if logistic is selected
py -m src.pressure
py -m src.ball_values
py -m src.stage4_ridge
py -m src.stage4_report
py -m pytest -q
```
