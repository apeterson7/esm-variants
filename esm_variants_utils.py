from typing import Tuple
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

class ProteinLanguageModelPredictor():

  def __init__(self, model_name, repr_layer=33, device=0, silent=False):
    self.AAorder=['K','R','H','E','D','N','Q','T','S','C','G','A','V','L','I','M','P','Y','F','W']
    self.device = device
    self.silent = silent
    self.repr_layer = repr_layer
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    self.batch_converter = alphabet.get_batch_converter() 
    self.model = model.eval().to(device)
    self.alphabet = alphabet

  def get_wt_LLR(self, input_df: pd.DataFrame) -> Tuple[list, list]: 
    # input: df.columns= id,	gene,	seq, length
    # requires global model and batch_converter
    # make sure input_df does not contain any nonstandard amino acids
    genes = input_df.id.values
    llr_dfs=[]
    for gene_name in tqdm(genes,disable=self.silent):
      input_sequence = input_df[input_df.id==gene_name].seq.values[0]
      input_sequence_length = len(input_sequence)
      
      if input_sequence_length<=1022:
        logits = self._get_logits(input_sequence, gene_name)
      else: ### tiling
        logits = self._get_logits_with_tiling(input_sequence, gene_name)

      llr_df = self._create_llr_dataframe(input_sequence, logits)
      llr_dfs.append(llr_df)

    return genes, llr_dfs

  def _get_logits(self, seq: str, gene_name: str = ""):
    data = [ (gene_name+"_", seq),]
    _, _, batch_tokens = self.batch_converter(data)
    batch_tokens = batch_tokens.to(self.device)
    with torch.no_grad():
      logits = torch.log_softmax(self.model(batch_tokens, repr_layers=[33], return_contacts=False)["logits"],dim=-1).cpu().numpy()
      return logits[0][1:-1,:]

  def _get_logits_with_tiling(self, input_sequence: str, gene_name: str):
    intervals, M_norm = self.get_intervals_and_weights(len(input_sequence),min_overlap=512,max_len=1022,s=20)
        
    dt = []
    for i,idx in enumerate(intervals):
      dt += [(gene_name+'_WT_'+str(i),''.join(list(np.array(list(input_sequence))[idx])) )]

    logit_parts = []
    for dt_ in self.chunks(dt,20):
      _, _, batch_tokens = self.batch_converter(dt_)
      with torch.no_grad():
        results_ = torch.log_softmax(self.model(batch_tokens.to(self.device), repr_layers=[33], return_contacts=False)['logits'],dim=-1)
      for i in range(results_.shape[0]):
        logit_parts += [results_[i,:,:].cpu().numpy()[1:-1,:]]
        
    logits_full = np.zeros((len(input_sequence),33))
    for i in range(len(intervals)):
      logit = np.zeros((len(input_sequence),33))
      logit[intervals[i]] = logit_parts[i].copy()
      logit = np.multiply(logit.T, M_norm[i,:]).T
      logits_full+=logit
    
    return logits_full

  def _create_llr_dataframe(self, input_sequence: str, logits) -> pd.DataFrame:
    wt_logits_df = pd.DataFrame(
      logits,
      columns=self.alphabet.all_toks,
      index=list(input_sequence)
    ).T.iloc[4:24].loc[self.AAorder]

    wt_logits_df.columns = [j.split('.')[0]+' '+str(i+1) for i,j in enumerate(wt_logits_df.columns)]
    wt_norm=np.diag(wt_logits_df.loc[[i.split(' ')[0] for i in wt_logits_df.columns]])
    llr_df = wt_logits_df - wt_norm
    return llr_df

  ##################### TILING utils ###########################

  def chunks(self, lst: list, n: int):
      """Yield successive n-sized chunks from lst."""
      for i in range(0, len(lst), n):
          yield lst[i:i + n]

  def chop(self, L,min_overlap=511,max_len=1022):
    return L[max_len-min_overlap:-max_len+min_overlap]
    
  def intervals(self, L,min_overlap=511,max_len=1022,parts=[]):
    #print('L:',len(L))
    #print(len(parts))
    if len(L)<=max_len:
      if parts[-2][-1]-parts[-1][0]<min_overlap:
        #print('DIFF:',parts[-2][-1]-parts[-1][0])
        return parts+[np.arange(L[int(len(L)/2)]-int(max_len/2),L[int(len(L)/2)]+int(max_len/2)) ]
      else:
        return parts
    else:
      parts+=[L[:max_len],L[-max_len:]]
      L=self.chop(L,min_overlap,max_len)
      return self.intervals(L,min_overlap,max_len,parts=parts)

  def get_intervals_and_weights(self, seq_len, min_overlap=512, max_len=1022, s=20):
    intervals = self.intervals(np.arange(seq_len), min_overlap=min_overlap, max_len=max_len)
    ## sort intervals
    intervals = [intervals[i] for i in np.argsort([i[0] for i in intervals])]

    a=int(np.round(min_overlap/2))
    t=np.arange(max_len)

    f=np.ones(max_len)
    f[:a] = 1 / (1 + np.exp(-(t[:a]-a/2)/s))
    f[max_len-a:] = 1 / (1 + np.exp((t[:a]-a/2)/s))

    f0=np.ones(max_len)
    f0[max_len-a:] = 1 / (1 + np.exp((t[:a]-a/2)/s))

    fn=np.ones(max_len)
    fn[:a] = 1 / (1 + np.exp(-(t[:a]-a/2)/s))

    filt=[f0]+[f for i in intervals[1:-1]]+[fn]
    M = np.zeros((len(intervals),seq_len))
    for k,i in enumerate(intervals):
      M[k,i] = filt[k]
    M_norm = M/M.sum(0)
    return (intervals, M_norm)

  
  ##################### Psuedo Log Likelihood ###########################

  def get_PLL(self, seq: str, reduce = np.sum):
    s=self._get_logits(seq)
    idx=[self.alphabet.tok_to_idx[i] for i in seq]
    return reduce(np.diag(s[:,idx])) 

  ## PLLR score for indels
  def get_PLLR(self,wt_seq,mut_seq,start_pos,weighted=False):
    fn=np.sum if not weighted else np.mean
    if max(len(wt_seq),len(mut_seq))<=1022:
      return self.get_PLL(mut_seq,fn) - self.get_PLL(wt_seq,fn)
    else:
      wt_seq,mut_seq,start_pos = self.crop_indel(wt_seq,mut_seq,start_pos)
      return self.get_PLL(mut_seq,fn) - self.get_PLL(wt_seq,fn)

  def crop_indel(self, ref_seq,alt_seq,ref_start):
    # Start pos: 1-indexed start position of variant
    left_pos = ref_start-1
    offset = len(ref_seq)-len(alt_seq)
    start_pos = int(left_pos - 1022 / 2)
    end_pos1 = int(left_pos + 1022 / 2) -min(start_pos,0)+ min(offset,0)
    end_pos2 = int(left_pos + 1022 / 2) -min(start_pos,0)- max(offset,0)
    if start_pos < 0: start_pos = 0 # Make sure the start position is not negative
    if end_pos1 > len(ref_seq): end_pos1 = len(ref_seq) # Make sure the end positions are not beyond the end of the sequence
    if end_pos2 > len(alt_seq): end_pos2 = len(alt_seq)
    if start_pos>0 and max(end_pos2,end_pos1) - start_pos <1022: ## extend to the left if there's space
              start_pos = max(0,max(end_pos2,end_pos1)-1022)

    return ref_seq[start_pos:end_pos1],alt_seq[start_pos:end_pos2],start_pos-ref_start

  ## stop gain variant score
  def get_minLLR(self, seq,stop_pos):
    return min(self.get_wt_LLR(pd.DataFrame([('_','_',seq,len(seq))],columns=['id','gene','seq','length'] ))[1][0].values[:,stop_pos:].reshape(-1))
  
  ##################### MELTed CSV ###########################

  def meltLLR(self, LLR,savedir=None):
    vars = LLR.melt(ignore_index=False)
    vars['variant'] = [''.join(i.split(' '))+j for i,j in zip(vars['variable'],vars.index)]
    vars['score'] = vars['value']
    vars = vars.set_index('variant')
    vars['pos'] = [int(i[1:-1]) for i in vars.index]
    del vars['variable'],vars['value']
    if savedir is not None:
        vars.to_csv(savedir+'var_scores.csv')
    return vars

# ############### EXAMLE ##################
# ## Load model
# plmPredictor = ProteinLanguageModelPredictor(model_name='esm1b_t33_650M_UR50S')
# ## Create a toy dataset
# df_in = pd.DataFrame([('P1','gene1','FISHWISHFQRCHIPSTHATARECRISP',28),
#                       ('P2','gene2','RAGEAGAINSTTHEMACHINE',21),
#                       ('P3','gene3','SHIPSSAILASFISHSWIM',19),
#                       ('P4','gene4','A'*1948,1948)], columns = ['id','gene','seq','length'])
# ## Get LLRs
# ids,LLRs = plmPredictor.get_wt_LLR(df_in)
# for i,LLR in zip(ids,LLRs):
#   print(i,LLR.shape)
# ## Get PLL
# print(plmPredictor.get_PLL(df_in.seq.values[0]))
# ## indel: 14_IPS_delins_EESE (FISHWISHFQRCHIPSTHATARECRISP --> FISHWISHFQRCHEESETHATARECRISP)
# plmPredictor.get_PLLR('FISHWISHFQRCHIPSTHATARECRISP','FISHWISHFQRCHEESETHATARECRISP',14)
# ## stop at position 17
# plmPredictor.get_minLLR(df_in.seq.values[0],17)