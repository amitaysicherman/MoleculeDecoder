import torch
import numpy as np

vocab = ['<pad>', '<unk>', '<bos>', '<eos>', '[Eu]', '[Db]', '[Nd]', '[BiH4-]', '%65', '[Hg+]', '[BH4-]', '%93',
         '[MnH+]', '[OsH-]', '[TiH+3]', '[SiH4]', '[Th]', '[Pd+]', '[PH2-]', '[SeH-]', '[C]', '[Zr-]', '[Pd+2]', '%64',
         '[Ni-]', '[BH3-]', '[Ar]', '[P+]', '%42', '[P-]', '4', '[Re-]', '[Fe+3]', '[AlH3-]', '[TiH+]', '[Ir-2]',
         '[Ge+4]', '[Al-3]', '%39', '[Fe-2]', '[ThH4]', '[ReH7]', '%46', '[SnH-]', '[I]', '3', '[GeH4]', '[Xe]', '[P]',
         '[Es]', '[PH4]', '[BH2]', '[Tc+6]', '[SnH3+]', '[CH2]', '[Sg]', '[Fe-]', '[Md]', '[N-]', '[Ta-]', '[Kr]',
         '%84', '[si-]', '[Hg-]', '[Os+7]', '[Mn+]', '[CoH+2]', '%28', '[c-]', '[GaH]', '[Hs]', '[HH]', '[Li]', '[Po]',
         '[Bi+]', '%78', '%95', '[K]', '[Os-2]', '[TeH3+]', '[Te]', '[Pb+2]', '[Cu-2]', '[La]', '[YH2]', '%94',
         '[SbH4+]', '[ClH2+3]', '[CoH]', '[Br-]', '[Nb+5]', 'n', '[PoH2]', '[Fe+4]', '.', '7', 'Br', '[Ta+]', '[RuH3]',
         '[Nb-2]', '[Fe+5]', '[Pt-]', '[S-]', '%97', '[SH6]', '[Np]', '[Os+8]', '[FeH6-3]', '[ThH2]', 'S', '[TiH4]',
         '[SeH2+]', '[Bi-]', '[Sm]', '%80', '[Dy]', '[ThH]', '%34', '[CH3]', '[Mn+2]', '[SnH2+]', '%(101)', '#',
         '[AuH]', '[V+]', '[RuH6]', '[Ir+]', '[LaH3]', '%11', '[Zn-]', '[Os-3]', '[Ne]', '[UH]', '[nH]', '[Ti]', '%74',
         '[AlH3]', '[GaH2]', '[O+]', '[SbH2]', '[SbH+]', '[Hg+2]', '[HgH2]', '%71', '[GeH3]', '[Gd]', '[Os+6]', '[W+2]',
         '[Cu+2]', '[Al-2]', '[Pr+]', '[SnH+]', '[SmH3]', '[Zr+3]', '[AlH+]', '[F-]', '[Ta+2]', '[SH4]', '[Mo]', '[F]',
         '[SnH2]', '[Pr+3]', '[K+]', '[Ir]', '[Co]', '[SH3+]', '[OH]', '[SeH+]', '[S]', '[Bi+2]', '[Se-]', '[se+]',
         '[Os+5]', '[OH+]', '[NbH2]', '[Cl-]', '[Nb]', '[BH]', '[Al+]', '[CH2-]', '[Mn+3]', '[ReH4]', 'c', '[V+4]',
         '[RuH]', '[MnH2]', '[OH3+]', '[MnH]', '[SeH3]', '[Fm]', '[Ru-]', '%89', '[MoH4]', '[PoH]', '[Sn+2]', '[CH]',
         '[nH+]', 'C', '[TeH]', '[pH]', '[PdH2]', '[SnH+3]', '[H-]', '[AsH5]', '[P-2]', '[PbH2]', '[V+2]', '[PbH]',
         '[Lr]', '[Am]', '[GeH]', '[BH-]', '[SiH2+]', '[Os+2]', '[Be+2]', '[Se]', '[IH3]', '[Cu+]', '[Ba]', '[si]',
         '[AlH-]', '%54', '%18', '[Rb]', 'o', '[O-]', '[U]', '[Sc+3]', '[Tl+3]', '[Ta-2]', '[NH2+]', '[Hf+2]', '%68',
         '[Ru+5]', '[Au]', '%41', '%52', '[Mg+2]', '[Cd-2]', '[Ca+]', '[Rf]', '[PH3]', '[ClH4+3]', '[Bh]', '%67',
         '[U+3]', '[Te+4]', '9', '[Cr-2]', '[BH2-]', 'B', '[ReH2]', '[RuH4]', '[Pt+]', '%70', '[N+]', '[TaH5]',
         '[ZrH2]', '[CrH2]', '[TeH2]', '%38', '[AsH2-]', '[Tb+3]', '[Mt]', '[FeH6-4]', '[Br+]', '[C+]', '[Gd+3]',
         '[Ni-2]', '[IH4-]', '[Ta+5]', '[bH-]', '1', '[SnH3]', '[CeH3]', '[S+]', '[Sb-]', '[AsH3]', '[Y+3]', '[Fe]',
         '[Cr+3]', '[Nd+3]', '[Al-]', '[Ru+3]', '%55', 'b', '%23', '[SiH]', '[TaH]', '[Os+]', '[PtH2]', '[Si+3]',
         '[In-]', '[IH2-]', '[Cr+6]', '[Ni+2]', '[Sn]', '[ClH+]', '[Ti+3]', '[U+2]', '[NiH]', '[Cr+5]', '[Ho]', '%75',
         '[Pd-]', '[N]', '[La+3]', '[Mn-2]', '[NH]', '[Rh+]', '%50', '%36', '[Rh+3]', '[AsH3+]', 'P', '[CH2+]', '8',
         '[Sn+3]', '[AlH+2]', '%16', '[Ra]', '[Zr+]', '[Tc]', '[YH3]', '%72', '[Lu]', '[Ni+3]', '[WH]', '[Ba+]',
         '[ZrH]', '[Ce+4]', '[AsH2+]', '[AsH4+]', '[FeH]', '[RuH-]', '[Yb+3]', '[Sc]', '%99', '[NbH3]', '[n-]', '%35',
         '[Ru+8]', '%53', '[Se-2]', '[No]', '[FeH4]', '[n+]', '[Sn+4]', '(', '[NiH2]', '[IrH3]', '[Si-]', '[CuH]',
         '[XeH]', '%45', '%32', '[Cm]', '[Cl]', '[Na]', '[Sr]', '[siH-]', '%85', '[Ru+4]', '[Na+]', '[Cr+2]', '[Ge]',
         '[Mg+]', '[RhH3]', '[IH]', '[Cl+3]', '%48', '[Ti-]', '[Cd-]', '[SbH4]', '[Cr-]', '[Cr]', '[O]', '[PbH3]',
         '[Ru+2]', '[Mo+]', '[Si+]', '[s+]', '[EuH3]', '[Ba+2]', '[Li-]', '[AlH2]', '[RuH+]', '[Tc+7]', '%10',
         '[RuH+2]', '[WH3]', '[Si]', '[Nb-]', '[WH6]', '[SiH4-]', '%90', '[Eu+2]', '[Yb]', '[AsH]', '[Ag]', '[Rn]',
         '[Zn+]', '[PtH3]', '[Rh-3]', '[SH-]', '[BiH4]', '[PbH4]', '%87', '[Zn-2]', '%69', '[W-]', '[TiH2]', '%27',
         '[SiH2]', '[PtH+]', '[Th+4]', '[Tm]', '[Tl+]', '[Tl]', 'F', '[Pd-2]', '[c+]', '[He]', '%43', '[SnH4]',
         '[Co+3]', '5', '%58', '[FeH2]', '[Yb+2]', '[CoH3]', '[MoH5]', '[Cr-3]', '[Re]', 'p', '[Co+]', '[IrH2]', '[At]',
         '[PdH+]', '[FeH4-]', '[te]', '[Hf+]', '[TeH4]', '[Rh]', '[NH4+]', '[TiH]', '%73', '[SnH2+2]', '[Zr-4]',
         '[Pt-2]', '[RuH2+2]', '[SiH+]', '%(100)', '%13', '[MoH3]', '[Mo+2]', '[Cf]', '[NH2-]', '[AsH4]', '[Ru-2]',
         '[Ru-4]', '[Zr-2]', '[PH+]', '%19', '%29', '%88', '[Sr+2]', '[ClH3+2]', '%49', '[GeH2]', '[Sb+]', '[VH4]',
         '%63', '[CH-]', '[cH+]', '[Nb+2]', '[H+]', '[CrH+2]', '[InH]', '[Fe+]', '[UH3]', '[Fe+2]', '[Pt+4]', '[SiH3+]',
         '[SiH-]', '[NH+]', '[Ga-]', '[SiH3]', '[Cs+]', '[Y]', '[SH3]', '[PH-2]', '[BiH2+2]', '[Tb]', '[CH3-]',
         '[ClH2+]', '[Ag-]', '%40', '[SbH2+]', '[Pu]', '[Si+2]', '[Rb+]', '[CuH2-]', '[p+]', '%91', '[Mn]', '[SeH]',
         '[Hg-2]', '[Sb]', '[Zr+2]', '[RhH+2]', '[Pr]', '[Pt+2]', '[Se+6]', '%20', '[In+3]', '[I-]', '[W+]', '[Pa]',
         '[As-]', '[PbH2+2]', '[Ru+6]', '[p-]', '[Cu]', '[Ca]', '[Cr+4]', '[Fe+6]', '[Bi]', '[HgH]', '6', '[Zr+4]',
         '[ReH3]', '[I+3]', '[Hg]', '[ClH3+3]', '[SnH3-]', '[ZrH2+2]', '%83', '[TlH]', '[b-]', '%25', '[ReH]', '[PH2+]',
         '[As+3]', '[c]', '[SbH3+]', '[Au-]', '[Al+2]', '%33', '[SeH2]', '%86', '[U+4]', '%62', '[Cd+2]', '[IH2+]',
         '[Pd]', '%60', '[Co-2]', '[PH]', '[Cl+]', '[As+5]', '[B]', '[Lu+3]', '[Zn]', '%14', '[PH3+]', '[Tc+5]',
         '[SbH5]', '[YbH2]', '[Ca+2]', '[Au+3]', '[Cl+2]', '[Au+]', '[Br+2]', '[oH+]', '[TeH2+]', ')', '[CuH2]',
         '[PH2]', '[SH]', '[Er+3]', '[Er]', '[PdH]', '[AsH2]', '[RuH+3]', '[Sn-]', '[H]', '[ClH+2]', '[IrH+2]',
         '[InH2]', '[TlH2]', '[Sm+3]', '[NH3+]', '[O-2]', '[Mo+4]', '[te+]', '[Os]', '[SH2+]', '[ZnH+]', '%31', '[Pb]',
         '[IH+]', '[Ho+3]', '%77', '[Te+]', '[Tb+4]', '%61', '[Br]', '[Rh+2]', '[RuH5]', '[Ir-3]', '[NH2]', '[Ru]',
         '[Ce+3]', '%37', '%82', '[OH-]', '[Ni+]', '[SbH]', '%47', '[SiH2-]', '[Zr]', '[CH3+]', '[V]', '[BiH2+]',
         '[SeH5]', '[RuH2]', '[Mg]', '[InH3]', '[Ru+]', '[B-]', '[Al]', '[BiH5]', '[Rh-2]', '[si+2]', '[Se+4]',
         '[Zr-3]', '%26', '[Al+3]', '[AtH]', '[Pm]', '[OsH2]', '[VH2]', '[Ti+2]', '[Li+]', '[SH2]', '[Ni-3]', '%(103)',
         '[Br+3]', '%(102)', '[ZnH2]', '[ZrH3]', '[Bk]', '[WH2]', '[TeH+]', '[Ru-3]', '[Fe-3]', '[SH+]', '[Tc+4]',
         '%56', '[Re-2]', '[MoH]', '[Ti+4]', '[PH4+]', '[Ga+3]', '[As+]', '[Co-3]', '[AsH-]', '[VH3]', '[AlH4-]', '%22',
         '[pH+]', '[TeH3]', '[Ir+2]', '[Rh-]', '[Ti-2]', '[OsH4]', '[Os+4]', '%81', '[Ir+3]', '[AlH2+]', '=', '[Cr+]',
         '[IH3-]', 'N', '[Hf]', '[BiH2]', '[Cu-]', '[Cs]', '%12', '[Ta]', '%(104)', '[OH2+]', '[Gd+2]', '[Sb+3]',
         '[Os-]', '[GaH3]', '[IrH]', '[Ni]', '[Si+4]', '[Bi-2]', '[BiH]', '[In]', '[Ac]', '[TaH3]', '[Co-]', '[AgH]',
         '[UH2]', '%96', '[Sb+5]', '%24', '[C-]', '[PtH4]', '[Ag+]', '[IH-]', '[B+]', '[AlH2-]', '%44', '[YH]', '[Ce]',
         '[Co+2]', '%15', '[OsH]', '[W]', '[RhH2]', '2', '[SnH+2]', '[Fe-4]', '[I+]', '[CoH+]', '[Fr]', '[Se+]',
         '[Tm+3]', '[TaH2]', '[Ga]', '[Sn+]', '[GaH4-]', '[XeH2]', '[I+2]', '[In+]', '[ZnH]', '[CH+]', '%79', '[se]',
         '%57', '[SeH3+]', '%66', '[Zn+2]', '[FeH3]', '[SiH3-]', '[Eu+3]', '[MoH2]', '[AsH+]', '[Cd]', '[Tc+3]',
         '[CuH+]', '[WH4]', '%30', '[Dy+3]', '[IH5]', '[NH-]', '[TlH3]', '[VH]', '%92', '[ClH2+2]', '[PH5]', '%98',
         '[Pt]', '[CoH2]', '[AlH6-3]', '[SiH+2]', '%17', '-', '[o+]', '[IrH4]', 'O', '[P-3]', '[SnH2-]', '%76',
         '[BiH3]', '[Tc+]', 'I', 'Cl', '[Tc+2]', '[S-2]', '%51', '[PtH2+2]', '[SbH3]', '[PH-]', '[Ti+]', '[As]', '%59',
         '[IH2]', '[Mo-2]', '%21', '[cH-]', '[AlH]', '[Nb+3]', '[Mo+3]', '[Be]', '[siH]', '[Hf+4]', '[SnH]', '[sH+]',
         '[AsH3-]', '[RhH]', '[Th+2]', '[Bi+3]', '[PtH]', 's']


class ChemicalTokenizer:
    def __init__(self, word_list=vocab):
        # Add special tokens
        self.vocab_size = len(word_list)
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.eos_token_id = word_list.index(self.eos_token)
        self.bos_token_id = word_list.index(self.bos_token)
        self.pad_token_id = word_list.index(self.pad_token)
        self.unk_token_id = word_list.index(self.unk_token)
        self.vocab = {token: idx for idx, token in enumerate(word_list)}
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        self.special_tokens = [self.eos_token, self.bos_token, self.pad_token, self.unk_token]

    def encode(self, text, max_length=75, add_bos_eos=True, return_tensors='pt', print_unk=True, raise_unks=False):
        tokens = text.split()
        if add_bos_eos:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        if print_unk or raise_unks:
            if self.unk_token_id in token_ids:
                # find all the indexes:
                unk_indexes = [i for i, x in enumerate(token_ids) if x == self.unk_token_id]
                unk_count = len(unk_indexes)
                unk_text = [tokens[i] for i in unk_indexes]
                if print_unk:
                    print(f"Found {unk_count} unknown tokens: {unk_text}")
                if raise_unks:
                    raise ValueError(f"Found {unk_count} unknown tokens: {unk_text}")
        n_tokens = len(token_ids)
        attention_mask = [1] * n_tokens
        token_to_pad = max_length - n_tokens
        if token_to_pad < 0:
            token_ids = token_ids[:max_length]
            attention_mask = attention_mask[:max_length]
        else:
            token_ids = token_ids + [self.pad_token_id] * token_to_pad
            attention_mask = attention_mask + [0] * token_to_pad
        return_dict = {
            "input_ids": token_ids,
            "attention_mask": attention_mask
        }
        if return_tensors == 'pt':
            return_dict = {k: torch.tensor(v) for k, v in return_dict.items()}
        elif return_tensors == 'np':
            return_dict = {k: np.array(v) for k, v in return_dict.items()}
        return return_dict

    def decode(self, token_ids: list, skip_special_tokens=True):
        # convert to list :
        if isinstance(token_ids, torch.Tensor) or isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        # check list of lists:
        if isinstance(token_ids[0], list):
            raise ValueError("List of lists not supported")
        text = [self.inv_vocab[token_id] for token_id in token_ids]
        if skip_special_tokens:
            text = [token for token in text if token not in self.special_tokens]
        return " ".join(text)


# Example usage
if __name__ == "__main__":
    tokenizer = ChemicalTokenizer()
    with open("USPTO/src-train.txt") as f:
        lines = f.read().splitlines()
    for line in lines:
        encoded = tokenizer.encode(line)
        print("Encoded:", encoded)
        decoded = tokenizer.decode(encoded)
        print("Decoded:", decoded)
