
# In-Context Style Transfer (ICST) - Conditional Molecule Generation Dataset

This dataset models sets of molecules from various molecular domains as sequences of SMILES strings, ideal for training models in conditional molecular generation.
Molecules are represented as SMILES and seperated by '.' symbol.

## Dataset Overview
- **Domains Included:**
  - **Natural-products** (Coconut dataset) [Coconut](https://coconut.naturalproducts.net/collections?q=)
  - **Kinase inhibitors, withdrawn, malaria, tool compounds, pathogen,
 NIH mechanistic, lopac, natural product-based probes and drugs,
 zinc tool, axon medchem, adooq bioactive, novartis chemogenetic,
 drug matrix, PROTACs, covalentIn db, DrugBank compounds, reframe,
 cayman bioactiveall from the Probes & Drugs portal** and others from Probes & Drugs portal (Skuta et al., 2017)
  - **Product molecules** from the [USPTO-50k reaction dataset](https://figshare.com/articles/dataset/USPTO-50K_raw_/25459573?file=45206101)
  - **ZINClick** (bio, diversity,  green, yellow, orange, and red) [ZINClick](https://pubs.acs.org/doi/10.1021/ci400529h) (Levré et al., 2018)
  - **Active molecules** from [MoleculeNet](http://moleculenet.ai/) (BACE, BBBP, Tox21, etc.)
  - **Active molecules** from [FS-MOL](https://github.com/microsoft/FS-Mol)  (Stanley et al., 2021) 
  - **Active molecules** from [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
  - **Active molecules** from the [BELKA challenge](https://www.kaggle.com/competitions/leash-BELKA)  (Quigley et al., 2024)
- **Size:** Up to 100,000 compounds per domain, totaling 249 domains and more than 4M molecules

## Data Structure
Molecules within each domain are serialized and concatenated, separated by the `.` token. During training, their order is randomized to enhance model robustness.

## Data Splits
- **Train:** 80% of domains
- **Validation:** 10% of domains
- **Test:** 10% of domains

Domains are sorted by character length in descending order.

## Full list of domains
corresponding to order in train + valid + test set.

|     | super_domain                           | set                                    | split   |    # molecules |   character-length |
|----:|:---------------------------------------|:---------------------------------------|:--------|-------:|-----------:|
|   0 | coconut-09-2024                        | coconut-09-2024                        | train   | 100000 |    8606486 |
|   1 | zinclick-red                           | zinclick-red                           | train   | 100000 |    8349179 |
|   2 | belka active                           | belka_active                           | train   | 100000 |    7485776 |
|   3 | zinclick-orange                        | zinclick-orange                        | train   | 100000 |    7320802 |
|   4 | zinclick-yellow                        | zinclick-yellow                        | train   | 100000 |    6302359 |
|   5 | active pubchem23                       | active_pubchem23_train_valid           | train   | 100000 |    5738628 |
|   6 | active pubchem                         | active_pubchem_307535                  | train   | 100000 |    5286903 |
|   7 | active pubchem                         | active_pubchem_205674                  | train   | 100000 |    5251874 |
|   8 | active pubchem                         | active_pubchem_205447                  | train   |  99999 |    5175616 |
|   9 | active pubchem                         | active_pubchem_194959                  | train   | 100000 |    5174646 |
|  10 | active pubchem                         | active_pubchem_185764                  | train   | 100000 |    5150784 |
|  11 | active FSMOL                           | active_FSMOL_train_valid               | train   | 100000 |    5145798 |
|  12 | active pubchem                         | active_pubchem_168720                  | train   | 100000 |    5145487 |
|  13 | active pubchem                         | active_pubchem_2323                    | train   | 100000 |    5117371 |
|  14 | active pubchem                         | active_pubchem_152673                  | train   | 100000 |    5116874 |
|  15 | active pubchem                         | active_pubchem_286584                  | train   |  99999 |    5092839 |
|  16 | active pubchem                         | active_pubchem_1737                    | train   | 100000 |    5069082 |
|  17 | active pubchem                         | active_pubchem_1817                    | train   | 100000 |    5046634 |
|  18 | active pubchem                         | active_pubchem_2183                    | train   | 100000 |    5042877 |
|  19 | active pubchem                         | active_pubchem_1274                    | train   | 100000 |    5024410 |
|  20 | active pubchem                         | active_pubchem_1902                    | train   | 100000 |    5021111 |
|  21 | active pubchem                         | active_pubchem_1692                    | train   | 100000 |    5019001 |
|  22 | active pubchem                         | active_pubchem_1130                    | train   |  99999 |    4978475 |
|  23 | active pubchem                         | active_pubchem_574                     | train   | 100000 |    4960696 |
|  24 | active pubchem                         | active_pubchem_967                     | train   | 100000 |    4955053 |
|  25 | active pubchem                         | active_pubchem_767                     | train   | 100000 |    4947741 |
|  26 | active pubchem                         | active_pubchem_616                     | train   | 100000 |    4945782 |
|  27 | active pubchem                         | active_pubchem_1258                    | train   | 100000 |    4941592 |
|  28 | active pubchem                         | active_pubchem_355158                  | train   |  98749 |    4773797 |
|  29 | active pubchem                         | active_pubchem_466                     | train   |  96879 |    4694825 |
|  30 | active pubchem                         | active_pubchem_535                     | train   |  95857 |    4647138 |
|  31 | active pubchem                         | active_pubchem_382                     | train   |  86741 |    4184444 |
|  32 | active pubchem                         | active_pubchem_521520                  | train   |  65387 |    3317548 |
|  33 | active pubchem                         | active_pubchem_9                       | train   |  53053 |    3083095 |
|  34 | active pubchem                         | active_pubchem_0                       | train   |  52701 |    3065864 |
|  35 | active pubchem                         | active_pubchem_54                      | train   |  52613 |    3057400 |
|  36 | active pubchem                         | active_pubchem_4                       | train   |  51485 |    3006532 |
|  37 | active pubchem                         | active_pubchem_3                       | train   |  51517 |    3002848 |
|  38 | active pubchem                         | active_pubchem_2                       | train   |  51507 |    3001787 |
|  39 | active pubchem                         | active_pubchem_375                     | train   |  61606 |    2955682 |
|  40 | active pubchem                         | active_pubchem_51                      | train   |  50220 |    2925794 |
|  41 | active pubchem                         | active_pubchem_1                       | train   |  49051 |    2855890 |
|  42 | active pubchem                         | active_pubchem_7                       | train   |  48499 |    2835234 |
|  43 | active pubchem                         | active_pubchem_62                      | train   |  47587 |    2769904 |
|  44 | active pubchem                         | active_pubchem_6                       | train   |  46989 |    2758739 |
|  45 | active pubchem                         | active_pubchem_463146                  | train   |  46397 |    2533965 |
|  46 | active pubchem                         | active_pubchem_152812                  | train   |  45651 |    2067849 |
|  47 | cayman bioactives                      | cayman_bioactives                      | train   |  17996 |    1264365 |
|  48 | active pubchem                         | active_pubchem_13                      | train   |  14474 |     803363 |
|  49 | active pubchem                         | active_pubchem_8                       | train   |  13314 |     733897 |
|  50 | USPTO 50k                              | USPTO_50k_rc_1                         | train   |  15055 |     707745 |
|  51 | reframe                                | reframe                                | train   |  12305 |     689776 |
|  52 | drug bank                              | drug_bank                              | train   |  11810 |     663724 |
|  53 | USPTO 50k                              | USPTO_50k_rc_2                         | train   |  11839 |     633110 |
|  54 | active pubchem                         | active_pubchem_5                       | train   |  10910 |     606186 |
|  55 | active pubchem                         | active_pubchem_455145                  | train   |   1401 |     595131 |
|  56 | active pubchem                         | active_pubchem_67                      | train   |  10712 |     594470 |
|  57 | active pubchem                         | active_pubchem_131439                  | train   |  10803 |     569533 |
|  58 | active pubchem                         | active_pubchem_637                     | train   |  10478 |     556324 |
|  59 | active pubchem                         | active_pubchem_1682                    | train   |   9293 |     470246 |
|  60 | active pubchem                         | active_pubchem_644                     | train   |   7888 |     451486 |
|  61 | covalentIn db                          | covalentIn_db                          | train   |   6690 |     405310 |
|  62 | protac                                 | protac                                 | train   |   3270 |     405033 |
|  63 | USPTO 50k                              | USPTO_50k_rc_6                         | train   |   8187 |     376533 |
|  64 | drug matrix                            | drug_matrix                            | train   |   7304 |     363962 |
|  65 | active pubchem                         | active_pubchem_239948                  | train   |   7459 |     310639 |
|  66 | active pubchem                         | active_pubchem_239908                  | train   |   7479 |     307923 |
|  67 | active pubchem                         | active_pubchem_150971                  | train   |   6196 |     279597 |
|  68 | zinclick-green                         | zinclick-green                         | train   |   6042 |     275778 |
|  69 | active pubchem                         | active_pubchem_307520                  | train   |   7395 |     273884 |
|  70 | active pubchem                         | active_pubchem_307518                  | train   |   7331 |     273513 |
|  71 | USPTO 50k                              | USPTO_50k_rc_3                         | train   |   5629 |     268242 |
|  72 | active pubchem                         | active_pubchem_307509                  | train   |   7107 |     263067 |
|  73 | active pubchem                         | active_pubchem_321931                  | train   |   4652 |     246669 |
|  74 | active pubchem                         | active_pubchem_240068                  | train   |   6636 |     245182 |
|  75 | active FSMOL                           | active_FSMOL_4870                      | train   |   4928 |     235712 |
|  76 | moleculenet/toxcast active             | moleculenet/toxcast_active             | train   |   6186 |     235387 |
|  77 | active pubchem                         | active_pubchem_240100                  | train   |   4252 |     234349 |
|  78 | active pubchem                         | active_pubchem_463132                  | train   |   4245 |     234068 |
|  79 | novartis chemogenetic                  | novartis_chemogenetic                  | train   |   4185 |     224662 |
|  80 | active pubchem                         | active_pubchem_286562                  | train   |   5680 |     211577 |
|  81 | active pubchem                         | active_pubchem_321947                  | train   |   5857 |     211030 |
|  82 | active FSMOL                           | active_FSMOL_929                       | train   |   4228 |     210022 |
|  83 | active FSMOL                           | active_FSMOL_2709                      | train   |   4416 |     205139 |
|  84 | active FSMOL                           | active_FSMOL_4977                      | train   |   4695 |     202207 |
|  85 | adooq bioactives                       | adooq_bioactives                       | train   |   3188 |     199469 |
|  86 | zinclick-diversity                     | zinclick-diversity                     | train   |   2500 |     189386 |
|  87 | active FSMOL                           | active_FSMOL_5134                      | train   |   3481 |     167829 |
|  88 | USPTO 50k                              | USPTO_50k_rc_7                         | train   |   4553 |     167429 |
|  89 | active FSMOL                           | active_FSMOL_3211                      | train   |   3279 |     159983 |
|  90 | moleculenet/tox21 10k                  | moleculenet/tox21_10k_active           | train   |   3722 |     158308 |
|  91 | active FSMOL                           | active_FSMOL_4700                      | train   |   3219 |     153675 |
|  92 | active FSMOL                           | active_FSMOL_4976                      | train   |   3284 |     151017 |
|  93 | active FSMOL                           | active_FSMOL_5130                      | train   |   2868 |     149211 |
|  94 | active pubchem                         | active_pubchem_205588                  | train   |   2377 |     147626 |
|  95 | active pubchem                         | active_pubchem_151013                  | train   |   2787 |     146289 |
|  96 | active FSMOL                           | active_FSMOL_1685                      | train   |   3109 |     144259 |
|  97 | active FSMOL                           | active_FSMOL_5131                      | train   |   3086 |     135682 |
|  98 | active pubchem                         | active_pubchem_157708                  | train   |   2467 |     129660 |
|  99 | active pubchem                         | active_pubchem_150982                  | train   |   2441 |     127864 |
| 100 | active pubchem                         | active_pubchem_180409                  | train   |   2321 |     126397 |
| 101 | moleculenet/tox21 active               | moleculenet/tox21_active               | train   |   2872 |     123036 |
| 102 | active pubchem                         | active_pubchem_272801                  | train   |   2448 |     121903 |
| 103 | active FSMOL                           | active_FSMOL_3925                      | train   |   1630 |     119729 |
| 104 | active pubchem                         | active_pubchem_354983                  | train   |   1816 |     117101 |
| 105 | active pubchem                         | active_pubchem_157755                  | train   |   2091 |     115139 |
| 106 | active pubchem                         | active_pubchem_1625                    | train   |   2046 |     113518 |
| 107 | active FSMOL                           | active_FSMOL_4874                      | train   |   2230 |     109076 |
| 108 | active FSMOL                           | active_FSMOL_770                       | train   |   2230 |     109076 |
| 109 | active pubchem                         | active_pubchem_152663                  | train   |   2294 |     107581 |
| 110 | axon medchem                           | axon_medchem                           | train   |   2069 |     103944 |
| 111 | moleculenet/sider active               | moleculenet/sider_active               | train   |   1427 |      99133 |
| 112 | active pubchem                         | active_pubchem_230821                  | train   |   1820 |      98988 |
| 113 | active pubchem                         | active_pubchem_131467                  | train   |   1883 |      97932 |
| 114 | active FSMOL                           | active_FSMOL_4820                      | train   |   2113 |      96274 |
| 115 | active pubchem                         | active_pubchem_230685                  | train   |   1692 |      96156 |
| 116 | active FSMOL                           | active_FSMOL_5102                      | train   |   2061 |      94044 |
| 117 | moleculenet/hiv active                 | moleculenet/hiv_active                 | train   |   1443 |      92121 |
| 118 | active FSMOL                           | active_FSMOL_641                       | train   |   1822 |      91943 |
| 119 | active FSMOL                           | active_FSMOL_635                       | train   |   1804 |      90919 |
| 120 | active pubchem                         | active_pubchem_1769                    | train   |   1720 |      89490 |
| 121 | moleculenet/clintox active             | moleculenet/clintox_active             | train   |   1459 |      86018 |
| 122 | active FSMOL                           | active_FSMOL_5124                      | train   |   1877 |      82901 |
| 123 | active pubchem                         | active_pubchem_439                     | train   |   1665 |      81921 |
| 124 | active pubchem                         | active_pubchem_255068                  | train   |   1514 |      77884 |
| 125 | active pubchem                         | active_pubchem_1536                    | train   |   1584 |      77369 |
| 126 | active pubchem                         | active_pubchem_185848                  | train   |   1431 |      73818 |
| 127 | active FSMOL                           | active_FSMOL_5061                      | train   |   1580 |      71954 |
| 128 | USPTO 50k                              | USPTO_50k_rc_9                         | train   |   1807 |      70200 |
| 129 | active pubchem                         | active_pubchem_354966                  | train   |   1181 |      69758 |
| 130 | active FSMOL                           | active_FSMOL_4400                      | train   |   1496 |      65900 |
| 131 | active pubchem                         | active_pubchem_1644                    | train   |   1233 |      65583 |
| 132 | active pubchem                         | active_pubchem_1233                    | train   |   1249 |      65145 |
| 133 | moleculenet/bbbp active                | moleculenet/bbbp_active                | train   |   1510 |      65068 |
| 134 | zinc tool                              | zinc_tool                              | train   |   1305 |      63333 |
| 135 | active pubchem                         | active_pubchem_500720                  | train   |   1118 |      62887 |
| 136 | active pubchem                         | active_pubchem_352395                  | train   |    770 |      62695 |
| 137 | active pubchem                         | active_pubchem_152731                  | train   |   1241 |      61932 |
| 138 | active pubchem                         | active_pubchem_104                     | train   |   1135 |      60893 |
| 139 | active pubchem                         | active_pubchem_205373                  | train   |   1149 |      60872 |
| 140 | active pubchem                         | active_pubchem_179978                  | train   |   1250 |      60119 |
| 141 | active FSMOL                           | active_FSMOL_5060                      | train   |   1466 |      58542 |
| 142 | Natural product-based probes and drugs | Natural product-based probes and drugs | train   |    500 |      56577 |
| 143 | active FSMOL                           | active_FSMOL_3614                      | train   |    765 |      55735 |
| 144 | active pubchem                         | active_pubchem_152687                  | train   |   1002 |      54549 |
| 145 | active FSMOL                           | active_FSMOL_1886                      | train   |   1166 |      54208 |
| 146 | active FSMOL                           | active_FSMOL_4018                      | train   |   1101 |      53399 |
| 147 | active FSMOL                           | active_FSMOL_587                       | train   |   1141 |      52237 |
| 148 | lopac                                  | lopac                                  | train   |   1266 |      51430 |
| 149 | active FSMOL                           | active_FSMOL_4972                      | valid   |   1127 |      51135 |
| 150 | active FSMOL                           | active_FSMOL_2739                      | valid   |   1079 |      49561 |
| 151 | active pubchem                         | active_pubchem_1573                    | valid   |    920 |      49322 |
| 152 | active FSMOL                           | active_FSMOL_799                       | valid   |   1009 |      49257 |
| 153 | active FSMOL                           | active_FSMOL_3507                      | valid   |   1060 |      49191 |
| 154 | moleculenet/bace c                     | moleculenet/bace_c_active              | valid   |    691 |      48718 |
| 155 | active FSMOL                           | active_FSMOL_4677                      | valid   |   1075 |      47932 |
| 156 | active FSMOL                           | active_FSMOL_4969                      | valid   |   1037 |      47662 |
| 157 | active FSMOL                           | active_FSMOL_4141                      | valid   |   1027 |      47360 |
| 158 | active FSMOL                           | active_FSMOL_4973                      | valid   |   1017 |      47041 |
| 159 | active FSMOL                           | active_FSMOL_4926                      | valid   |    999 |      46585 |
| 160 | active FSMOL                           | active_FSMOL_4671                      | valid   |   1001 |      46548 |
| 161 | active pubchem                         | active_pubchem_2227                    | valid   |    900 |      46483 |
| 162 | active pubchem                         | active_pubchem_218098                  | valid   |    726 |      46390 |
| 163 | active pubchem                         | active_pubchem_240283                  | valid   |    816 |      46221 |
| 164 | active pubchem                         | active_pubchem_321701                  | valid   |    736 |      46117 |
| 165 | active FSMOL                           | active_FSMOL_5058                      | valid   |    985 |      45744 |
| 166 | active FSMOL                           | active_FSMOL_4674                      | valid   |    983 |      45452 |
| 167 | active FSMOL                           | active_FSMOL_4673                      | valid   |    939 |      45406 |
| 168 | active FSMOL                           | active_FSMOL_4669                      | valid   |    939 |      45406 |
| 169 | active FSMOL                           | active_FSMOL_4672                      | valid   |    989 |      44678 |
| 170 | active pubchem                         | active_pubchem_320688                  | valid   |    719 |      44412 |
| 171 | active FSMOL                           | active_FSMOL_312                       | valid   |    959 |      44391 |
| 172 | active FSMOL                           | active_FSMOL_3330                      | valid   |    950 |      44270 |
| 173 | active FSMOL                           | active_FSMOL_2042                      | valid   |    940 |      43896 |
| 174 | active FSMOL                           | active_FSMOL_4258                      | valid   |    958 |      43632 |
| 175 | active FSMOL                           | active_FSMOL_3757                      | valid   |    920 |      43305 |
| 176 | active FSMOL                           | active_FSMOL_2919                      | valid   |    938 |      43158 |
| 177 | active FSMOL                           | active_FSMOL_5068                      | valid   |    913 |      42972 |
| 178 | active FSMOL                           | active_FSMOL_979                       | valid   |    909 |      42481 |
| 179 | active FSMOL                           | active_FSMOL_5069                      | valid   |    921 |      42437 |
| 180 | active FSMOL                           | active_FSMOL_2558                      | valid   |    898 |      42032 |
| 181 | active FSMOL                           | active_FSMOL_3863                      | valid   |    892 |      41807 |
| 182 | active FSMOL                           | active_FSMOL_3692                      | valid   |    893 |      41604 |
| 183 | active FSMOL                           | active_FSMOL_892                       | valid   |    876 |      41443 |
| 184 | active FSMOL                           | active_FSMOL_5055                      | valid   |    884 |      41285 |
| 185 | active pubchem                         | active_pubchem_352438                  | valid   |    604 |      40203 |
| 186 | active FSMOL                           | active_FSMOL_605                       | valid   |    722 |      40017 |
| 187 | active FSMOL                           | active_FSMOL_5046                      | valid   |    853 |      39824 |
| 188 | active FSMOL                           | active_FSMOL_636                       | valid   |    889 |      39282 |
| 189 | active FSMOL                           | active_FSMOL_632                       | valid   |    836 |      39215 |
| 190 | active FSMOL                           | active_FSMOL_2137                      | valid   |    889 |      39210 |
| 191 | nih mechanistic                        | nih_mechanistic                        | valid   |    768 |      38606 |
| 192 | active FSMOL                           | active_FSMOL_1007                      | valid   |    809 |      38535 |
| 193 | USPTO 50k                              | USPTO_50k_rc_4                         | valid   |    898 |      38516 |
| 194 | active FSMOL                           | active_FSMOL_4314                      | valid   |    633 |      38347 |
| 195 | active FSMOL                           | active_FSMOL_3150                      | valid   |    632 |      38277 |
| 196 | active pubchem                         | active_pubchem_355032                  | valid   |    549 |      38216 |
| 197 | active FSMOL                           | active_FSMOL_1640                      | valid   |    791 |      38084 |
| 198 | active FSMOL                           | active_FSMOL_4869                      | valid   |    791 |      38084 |
| 199 | active FSMOL                           | active_FSMOL_4203                      | test    |    791 |      38084 |
| 200 | zinclick-bio                           | zinclick-bio                           | test    |    571 |      37977 |
| 201 | active FSMOL                           | active_FSMOL_3537                      | test    |    621 |      37530 |
| 202 | active pubchem                         | active_pubchem_578                     | test    |    697 |      37114 |
| 203 | active FSMOL                           | active_FSMOL_4521                      | test    |    813 |      36776 |
| 204 | active FSMOL                           | active_FSMOL_4646                      | test    |    813 |      36776 |
| 205 | active FSMOL                           | active_FSMOL_3613                      | test    |    813 |      36776 |
| 206 | active FSMOL                           | active_FSMOL_4254                      | test    |    607 |      36705 |
| 207 | active FSMOL                           | active_FSMOL_5032                      | test    |    760 |      35912 |
| 208 | active FSMOL                           | active_FSMOL_1916                      | test    |    999 |      35138 |
| 209 | active FSMOL                           | active_FSMOL_3051                      | test    |    768 |      35014 |
| 210 | tool compounds                         | tool_compounds                         | test    |    515 |      34802 |
| 211 | active FSMOL                           | active_FSMOL_1037                      | test    |    739 |      34787 |
| 212 | active FSMOL                           | active_FSMOL_902                       | test    |    735 |      34627 |
| 213 | active FSMOL                           | active_FSMOL_4020                      | test    |    618 |      34440 |
| 214 | active FSMOL                           | active_FSMOL_5040                      | test    |    851 |      34327 |
| 215 | active pubchem                         | active_pubchem_230754                  | test    |    680 |      34296 |
| 216 | malaria                                | malaria                                | test    |    816 |      34009 |
| 217 | USPTO 50k                              | USPTO_50k_rc_8                         | test    |    801 |      33655 |
| 218 | withdrawn 2                            | withdrawn_2                            | test    |    633 |      32990 |
| 219 | active FSMOL                           | active_FSMOL_514                       | test    |    685 |      32780 |
| 220 | active FSMOL                           | active_FSMOL_4971                      | test    |    649 |      32641 |
| 221 | active pubchem                         | active_pubchem_305650                  | test    |    471 |      32414 |
| 222 | active FSMOL                           | active_FSMOL_2107                      | test    |    690 |      32335 |
| 223 | active FSMOL                           | active_FSMOL_1762                      | test    |    683 |      32302 |
| 224 | active FSMOL                           | active_FSMOL_5042                      | test    |    667 |      31616 |
| 225 | active FSMOL                           | active_FSMOL_3075                      | test    |    647 |      30797 |
| 226 | active FSMOL                           | active_FSMOL_5008                      | test    |    754 |      30630 |
| 227 | active FSMOL                           | active_FSMOL_1462                      | test    |    694 |      29908 |
| 228 | active pubchem                         | active_pubchem_322401                  | test    |    387 |      29838 |
| 229 | active FSMOL                           | active_FSMOL_5133                      | test    |    662 |      29800 |
| 230 | active FSMOL                           | active_FSMOL_1946                      | test    |    623 |      29574 |
| 231 | active pubchem                         | active_pubchem_230747                  | test    |    534 |      29436 |
| 232 | active FSMOL                           | active_FSMOL_2889                      | test    |    684 |      29328 |
| 233 | active FSMOL                           | active_FSMOL_5020                      | test    |    609 |      28604 |
| 234 | active FSMOL                           | active_FSMOL_2020                      | test    |    615 |      28583 |
| 235 | active pubchem                         | active_pubchem_230680                  | test    |    493 |      28412 |
| 236 | active FSMOL                           | active_FSMOL_3990                      | test    |    658 |      28211 |
| 237 | active FSMOL                           | active_FSMOL_1876                      | test    |    704 |      27161 |
| 238 | USPTO 50k                              | USPTO_50k_rc_5                         | test    |    649 |      26460 |
| 239 | active FSMOL                           | active_FSMOL_4009                      | test    |    649 |      25143 |
| 240 | active pubchem                         | active_pubchem_321523                  | test    |    382 |      24803 |
| 241 | active pubchem                         | active_pubchem_321690                  | test    |    365 |      21834 |
| 242 | active pubchem                         | active_pubchem_321254                  | test    |    368 |      20732 |
| 243 | pathogen                               | pathogen                               | test    |    400 |      18772 |
| 244 | active pubchem                         | active_pubchem_352708                  | test    |    302 |      17516 |
| 245 | active pubchem                         | active_pubchem_179977                  | test    |    327 |      16464 |
| 246 | USPTO 50k                              | USPTO_50k_rc_10                        | test    |    226 |       7425 |
| 247 | kinase inhibitors                      | kinase_inhibitors                      | test    |     96 |       5867 |
| 248 | active pubchem                         | active_pubchem_92                      | test    |     75 |       5519 |
