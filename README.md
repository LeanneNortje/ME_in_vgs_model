# Using pretrained model checkpoints

If you want to use the model checkpoints, download the checkpoints given in the release and extract the checkpoints in the appropriate model folders.
<!---The following table gives the names in the releases that corresponds to the names in the ```model_metadata``` folder.-->
The next two tables contain more information. 
Take care to follow the exact directory layout given here:

```bash
├── model_metadata
│   ├── <model_name>
│   │   ├── <model_instance>
│   │   │   ├── models
│   │   │   ├── args.pkl
│   │   │   ├── params.json
│   │   │   ├── training_metadata.json
```

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Self-supervised AlexNet warmstart</th>
      <th>Self-supervised CPC warmstart</th>
      <th>Model name`</th>
      <th>Release name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">MattNet</td>
      <td><ul><li>- [ ] </li></ul></td>
      <td><ul><li>- [ ] </li></ul></td>
      <td>a35ce3ce67</td>
      <td>English_a35ce3ce67_1</td>
    </tr>
    <tr>
      <td><ul><li>- [x] </li></ul></td>
      <td><ul><li>- [ ] </li></ul></td>
      <td>4294a4933e</td>
      <td>English_4294a4933e_1</td>
    </tr>
    <tr>
      <td><ul><li>- [ ] </li></ul></td>
      <td><ul><li>- [x] </li></ul></td>
      <td>601394a62b</td>
      <td>English_601394a62b_1</td>
    </tr>
    <tr>
      <td><ul><li>- [x] </li></ul></td>
      <td><ul><li>- [x] </li></ul></td>
      <td>f613809f5c</td>
      <td>English_f613809f5c_1, English_f613809f5c_2, English_f613809f5c_3, English_f613809f5c_4, English_f613809f5c_5</td>
    </tr>
    <tr>
      <td>MattNet using a hinge loss</td>td
      <td><ul><li>- [x] </li></ul></td>
      <td><ul><li>- [x] </li></ul></td>
      <td>f613809f5c</td>
      <td>English_hinge_loss_f613809f5c_1</td>
    </tr>
    <tr>
      <td>MattNet using the InfoNCE loss</td>td
      <td><ul><li>- [x] </li></ul></td>
      <td><ul><li>- [x] </li></ul></td>
      <td>b4b77a981b</td>
       <td>English_infonce_b4b77a981b_1</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Supervised AlexNet warmstart</th>
      <th>Self-supervised CPC warmstart</th>
      <th>Model name</th>
      <th>Release name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MattNet</td>
      <td><ul><li>- [x] </li></ul></td>
      <td><ul><li>- [x] </li></ul></td>
      <td>44052e45b0</td>
      <td>English_supervised_alexnet_44052e45b0_1</td>
    </tr>
  </tbody>
</table>

# Data

# Preprocessing

```
cd preprocessing/
python preprocess_english_dataset.py
cd ../
```

# Model training

To run a new model:

```
python run.py
```

To resume training:

```
python run.py --resume
```


To resume training from a specific epoch:

```
python run.py --resume --restore-epoch <epoch_you_want_to_restore_from_minus_one>
```
For example, to restore from epoch 8, run:

```
python run.py --resume --restore-epoch 7
```

# Model testing

To evaluate familiar-<ins>novel</ins>
```
python test_ME.py
```
