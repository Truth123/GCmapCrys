import os
import math
import numpy as np
import re
from Bio import SeqIO
import sys
from string import Template
import pconsc4
import csv
import json
import h5py


projectPath = os.path.dirname(__file__)

## Third party execution tool path, need to be modified according to the actual path
PSI_BLAST_EXE = "pkgs/ncbi-blast-2.13.0+/bin/psiblast"
RUN_SCRATCH_EXE = "pkgs/SCRATCH-1D_1.2/bin/get_abinitioo_predictions.sh" 
HHBLITS_EXE = "pkgs/hhsuite/bin/hhblits"  
HHBLITSDB = "uniclust30_2018_08" 
MEFF_FILTER_EXE = "pkgs/Meff_Filter/meff_filter"
PI_EXE = "pkgs/ipc-2.0/scripts/ipc2_protein_svr_predictor.py"
PI_MODEL = "pkgs/ipc-2.0/models/IPC2_protein_75_SVR_19.pickle"


hhblits_template=Template("$hhblits -i $infile -diff inf -d $db -cpu $ncpu -oa3m $outprefix.a3m "+ \
" -id $id_cut -cov $cov_cut -o $outprefix.log -n 3; "+ \
"grep -v '^>' $outprefix.a3m|sed 's/[a-z]//g' > $outprefix.aln")

acid_index = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8,   
        'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 
        'Y': 18, 'V': 19, 'X': 20}
length_mean = 5.6669 
length_std = 0.4686
pi_mean = 6.4127
pi_std = 1.4838

def command_pssm(query_file, out_file, pssm_file, evalue):
    os.system('%s -query %s -db swissprot -evalue %f -num_iterations 3 -num_threads 10 -out %s -out_ascii_pssm %s'%(PSI_BLAST_EXE, query_file, evalue, out_file, pssm_file))

def command_scratch(query_file, output_prefix):
    os.system('%s %s %s %d'%(RUN_SCRATCH_EXE, query_file, output_prefix))

def get_pssm(id, seq):
    tmp_dir = os.path.join(projectPath, "tmp_pssm")
    os.mkdir(tmp_dir)
    query_fasta = os.path.join(tmp_dir, id + '.fasta')
    with open(query_fasta, 'w') as f:
        f.write('>'+id+'\n'+seq)
    out_file = os.path.join(tmp_dir, id + '.out')
    pssm_file = os.path.join(tmp_dir, id + '.pssm')

    command_pssm(query_fasta, out_file, pssm_file, 0.001)

    if not os.path.exists(pssm_file):
        os.system("rm -rf %s"%(tmp_dir))
        return None
    else:
        pssm = []
        with open(pssm_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = re.sub(r"\s{2,}", " ", line)
            if line != '' and line != '\n' and re.match(r'^\d*', line).group(0) != '':
                line = line.split(' ')[2:22]
                temp = [float(x) for x in line]
                pssm.append(temp)
        normal_pssm = []
        for row in pssm:
            normal_pssm.append([(1.0 / (1 + math.exp(-x))) for x in row])
        normal_pssm = np.array(normal_pssm)

        os.system("rm -rf %s"%(tmp_dir))
        return normal_pssm


def parse_rsa_acc(acc_file, type='acc20'):
    with open(acc_file,'r') as f:
        lines = f.readlines()
    id = ''
    status = 'head'
    value = []
    data = {}
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if status == 'acc':
                new_value = []
                for i, x in enumerate(value):
                    if x == "-":
                        new_value.append(0)
                    elif x == "e":
                        new_value.append(1)
                    else:
                        new_value.append(float(x))
                data[id] =new_value
                status = 'head'
                value = []
            id = line[1:]
            status = 'acc'
            continue
        if line != '' and status == 'acc':
            if type == "acc20":
                line = re.sub('\s+',' ',line)
                value += line.split(' ')
            else:
                value = line
    if status == 'acc' and value != []:
        new_value = []
        for i, x in enumerate(value):
            if x == "-":
                new_value.append(0)
            elif x == "e":
                new_value.append(1)
            else:
                new_value.append(float(x))
        data[id] =new_value
    return data


def parse_ss8(ss8_file):
    ss_index = {
        'G': 0, 'H': 1, 'I': 2, 'E': 3, 'B': 4, 'S': 5, 'T': 6, 'C': 7
    }
    data = {}
    for it in SeqIO.parse(ss8_file, 'fasta'):
        id = it.id
        row = str(it.seq)
        row = [ss_index[x] for x in row]
        data[id] = row
    return data


def get_scratch(id, seq):
    tmp_dir = os.path.join(projectPath, "tmp_scratch")
    os.mkdir(tmp_dir)
    query_fasta = os.path.join(tmp_dir, id + '.fasta')
    with open(query_fasta, 'w') as f:
        f.write('>'+id+'\n'+seq)
    
    output_prefix = os.path.join(tmp_dir, id)
    command_scratch(query_fasta, output_prefix)

    out_ss = output_prefix + '.ss'
    out_ss8 = output_prefix + '.ss8'
    out_acc = output_prefix + '.acc'
    out_acc20 = output_prefix + '.acc20'

    if not os.path.exists(out_ss) or\
        not os.path.exists(out_ss8) or\
        not os.path.exists(out_acc) or\
        not os.path.exists(out_acc20):
        os.system("rm -rf %s"%(tmp_dir))
        return None
    else:
        ss8 = parse_ss8(out_ss8)
        ss8 = np.array(ss8[id])
        acc20 = parse_rsa_acc(out_acc20)
        acc20 = np.array(acc20[id]) / 100.0
        os.system("rm -rf %s"%(tmp_dir))
        return (ss8, acc20)


def get_ccmap(id, seq):
    tmp_dir = os.path.join(projectPath, "tmp_ccmap")
    os.mkdir(tmp_dir)

    prefix = os.path.join(tmp_dir, id)
    fasta_file = os.path.join(tmp_dir, id + '.fasta')
    with open(fasta_file, 'w') as f:
        f.write('>'+id+'\n'+seq)

    a3m_file = prefix+'.a3m'
    a3m_file_filter = prefix+'.a3m_filter'
    aln_file = prefix+'.aln'
    log_file = prefix+'.log'

    hhblits_cmd=hhblits_template.substitute(
        hhblits   = HHBLITS_EXE,
        infile    = fasta_file,
        db        = HHBLITSDB,
        ncpu      = 4,
        outprefix = prefix,
        id_cut    = 99,
        cov_cut   = 50,
        )
    os.system(hhblits_cmd)
    if not os.path.isfile(a3m_file) or not os.path.isfile(aln_file) or not os.path.isfile(log_file):
        sys.stderr.write("ERROR! run hhsuite failed")
        os.system("rm -rf %s"%(tmp_dir))
        return None

    with open(a3m_file,'r') as f:
        lines = f.readlines()
    num_msa = len(lines) / 2
    if num_msa > 50000:
        filter_command = "%s -i %s -o %s -s 0.99 -n %d -d 1 -v 1"%(MEFF_FILTER_EXE, a3m_file, a3m_file_filter, 50000)
        os.system(filter_command)
        os.system("rm -rf %s"%(a3m_file))
        os.system("mv %s %s"%(a3m_file_filter, a3m_file))

    model = pconsc4.get_pconsc4()
    try:
        results = pconsc4.predict_contacts(model, a3m_file,verbose=2)
    except:
        print("pconsc4 predict %s failed"%(a3m_file))
        return None
    ccmap = np.array(results['cmap'])
    edge_index = []
    edge_attr = []
    seq_len = ccmap.shape[0]
    for i in range(seq_len):
        for j in range(seq_len):
            if i != j and ccmap[i][j] >= 0.3:
                edge_index.append([i,j])
                edge_attr.append([ccmap[i][j]])
    edge_index = np.asarray(edge_index).T
    edge_attr = np.asarray(edge_attr)

    os.system("rm -rf %s"%(a3m_file))
    return (edge_index, edge_attr)


def get_pI(id, seq):
    tmp_dir = os.path.join(projectPath, "tmp_pI")
    os.mkdir(tmp_dir)
    fasta_file = os.path.join(tmp_dir, id + '.fasta')
    with open(fasta_file, 'w') as f:
        f.write('>'+id+'\n'+seq)
    result_file = os.path.join(tmp_dir, id + '.csv')
    os.system("python %s %s %s %s"%(PI_EXE, PI_MODEL, fasta_file, result_file))
    if not os.path.isfile(result_file):
        os.system("rm -rf %s"%(tmp_dir))
        return None
    with open(result_file, 'r') as f:
        f_csv = csv.reader(f)
        header = next(f_csv)
        value = float(header[0].split(':')[-1].strip())
        os.system("rm -rf %s"%(tmp_dir))
        return value

def get_aaindex_gravy(seq):
    AAindex_file = os.path.join(projectPath, "aaindex_scale.json")
    with open(AAindex_file, 'r') as f:
        all_aaindex_dict = dict(json.load(f))
    aaindex_scale = []
    Gravy = 0

    aa = []
    for x in seq:
        ## X
        if x not in acid_index.keys():
            aa.append[20]
        ## 20 standard acid
        else:
            aa.append(acid_index[x])
    aa = np.array(aa)
    
    for entry in all_aaindex_dict.keys():
        index = all_aaindex_dict[entry]
        l = []
        for n in aa:
            ## X
            if n == 20:
                l.append(0)
            ## 20 standard acid
            else:
                l.append(index[n])
        aaindex_scale.append(l)
        if entry == 'KYTJ820101':
            Gravy = np.mean(l)
    aaindex_scale = np.array(aaindex_scale)
    return aaindex_scale, Gravy


def generate_feats(data, out_dir):

    for id in data.keys():
        seq = data[id]

        ## 1. run psiblast
        pssm = get_pssm(id, seq)
        if pssm is None:
            print("run psiblast for %s failed"%(id))
            continue
        
        ## 2. run scratch
        scratch = get_scratch(id, seq)
        if scratch is None:
            print("run scratch for %s failed"%(id))
            continue
        else:
            ss8, acc20 = scratch

        ## 3. run pconsc4 and get ccmap
        ccmap = get_ccmap(id ,seq)
        if ccmap is None:
            print("run hhsuite and pconsc4 for %s failed"%(id))
            continue
        else:
            edge_index, edge_attr = ccmap

        ## 4. get pI
        pi_value = get_pI(id, seq)
        if pi_value is None:
            print("run ipc2.0 for %s failed"%(id))
            continue
        pi_scale = (pi_value - pi_mean) / pi_std

        ## 6. get AAindex and Gravy
        aaindex_scale, Gravy = get_aaindex_gravy(seq)

        ## 7. log_length

        length = len(seq)
        log_length = (np.log(length) - length_mean) / length_std

        ## 8. AA
        aa = []
        for x in seq:
            ## X
            if x not in acid_index.keys():
                aa.append[20]
            ## 20 standard acid
            else:
                aa.append(acid_index[x])
        aa = np.array(aa)

        f = h5py.File(os.path.join(out_dir, id+'.h5'), 'w')
        d = f.create_dataset("id", data=id)
        d = f.create_dataset("seq", data=seq)
        d = f.create_dataset("log_length", data=log_length)
        d = f.create_dataset("pI", data=pi_scale)
        d = f.create_dataset("Gravy", data=Gravy)
        d = f.create_dataset("AA", data=aa)
        d = f.create_dataset("edge_index", data=edge_index)
        d = f.create_dataset("edge_attr", data=edge_attr)
        d = f.create_dataset("PSSM", data=pssm)
        d = f.create_dataset("AAindex", data=aaindex_scale)
        d = f.create_dataset("RSA", data=acc20)
        d = f.create_dataset("SS", data=ss8)
        f.close()
        
    

    