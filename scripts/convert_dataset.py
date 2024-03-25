# %%
import os
import pandas as pd
import numpy as np
import awkward
import uproot_methods
import sys
sys.path.append('..')
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='data/test.h5', help='Path to the source file')
parser.add_argument('--destination', type=str, default='data/converted', help='Path to the destination directory')
parser.add_argument('--type', type=str, default='test', help='Type of the dataset')
args = parser.parse_args()

if not os.path.exists(args.destination):
    os.makedirs(args.destination)

def _transform(dataframe, start=0, stop=-1, jet_size=0.8):
    from collections import OrderedDict
    v = OrderedDict()

    df = dataframe.iloc[start:stop]
    def _col_list(prefix, max_particles=200):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]
    
    _px = df[_col_list('PX')].values
    _py = df[_col_list('PY')].values
    _pz = df[_col_list('PZ')].values
    _e = df[_col_list('E')].values
    
    mask = _e>0
    n_particles = np.sum(mask, axis=1)

    px = awkward.JaggedArray.fromcounts(n_particles, _px[mask])
    py = awkward.JaggedArray.fromcounts(n_particles, _py[mask])
    pz = awkward.JaggedArray.fromcounts(n_particles, _pz[mask])
    energy = awkward.JaggedArray.fromcounts(n_particles, _e[mask])

    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(px, py, pz, energy)
    pt = p4.pt

    jet_p4 = p4.sum()

    # outputs
    _label = df['is_signal_new'].values
    v['label'] = np.stack((_label, 1-_label), axis=-1)
    v['train_val_test'] = df['ttv'].values
    
    v['jet_pt'] = jet_p4.pt
    v['jet_eta'] = jet_p4.eta
    v['jet_phi'] = jet_p4.phi
    v['jet_mass'] = jet_p4.mass
    v['n_parts'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    v['part_pt_log'] = np.log(pt)
    v['part_ptrel'] = pt/v['jet_pt']
    v['part_logptrel'] = np.log(v['part_ptrel'])

    v['part_e_log'] = np.log(energy)
    v['part_erel'] = energy/jet_p4.energy
    v['part_logerel'] = np.log(v['part_erel'])

    v['part_raw_etarel'] = (p4.eta - v['jet_eta'])
    _jet_etasign = np.sign(v['jet_eta'])
    _jet_etasign[_jet_etasign==0] = 1
    v['part_etarel'] = v['part_raw_etarel'] * _jet_etasign

    v['part_phirel'] = p4.delta_phi(jet_p4)
    v['part_deltaR'] = np.hypot(v['part_etarel'], v['part_phirel'])

    v['part_pt_log'] = (v['part_pt_log'] - 1.7) * 0.7
    v['part_e_log'] = (v['part_e_log'] - 2.0) * 0.7
    v['part_logptrel'] = (v['part_logptrel'] - (-4.7)) * 0.7
    v['part_logerel'] = (v['part_logerel'] - (-4.7)) * 0.7
    v['part_deltaR'] = (v['part_deltaR'] - 0.2) * 4.0


    return v

# %%
def convert(source, step=None, limit=None):
    df = pd.read_hdf(source, key='table')
  #  logging.info('Total events: %s' % str(df.shape[0]))
    if limit is not None:
        df = df.iloc[0:limit]
     #   logging.info('Restricting to the first %s events:' % str(df.shape[0]))
    if step is None:
        step = df.shape[0]
    idx=-1
    while True:
        idx+=1
        start=idx*step
        if start>=df.shape[0]: break
       # if not os.path.exists(destdir):
        #    os.makedirs(destdir)
        # output = os.path.join(destdir, '%s_%d.awkd'%(basename, idx))
        #logging.info(output)
        #if os.path.exists(output):
         #   logging.warning('... file already exist: continue ...')
          #  continue
        v=_transform(df, start=start, stop=start+step)
    return v
       # awkward.save(output, v, mode='x')

srcDir = 'original'
destDir = 'converted'

v_converted = convert(args.source)
#test = np.array(v_converted['part_pt_log'])

def pad_array(array, pad_to=128, constant_value=0):
    if array.shape[0] > pad_to:
        return array[:pad_to]
    else:
        return np.pad(array, ((0, pad_to - array.shape[0])), mode='constant', constant_values=constant_value)


def create_padded_array(array, feature_list=['part_pt_log',
                                             'part_e_log', 
                                             'part_logptrel', 
                                             'part_logerel', 
                                             'part_deltaR', 
                                             'part_etarel', 
                                             'part_phirel']):
    
    features = [np.array(array[feature]) for feature in feature_list]
    stacked_padded_features = []

    for feature_index in range(len(features)):
        padded_features = np.array([np.array(pad_array(features[feature_index][event])).reshape(-1,1) for event in range(len(features[feature_index]))])
        stacked_padded_features.append(padded_features)

    stacked_padded_features = np.concatenate(stacked_padded_features, axis=-1)
    return stacked_padded_features


part_features = create_padded_array(v_converted)
part_4momenta = create_padded_array(v_converted, feature_list=['part_px', 'part_py', 'part_pz', 'part_energy'])
part_labels = v_converted['label']

np.save(f'{args.destination}/{args.type}_part_features.npy', part_features)
np.save(f'{args.destination}/{args.type}_part_4momenta.npy', part_4momenta)
np.save(f'{args.destination}/{args.type}_part_labels.npy', part_labels)



