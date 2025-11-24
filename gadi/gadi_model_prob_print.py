#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import tempfile
import pandas as pd
import numpy as np
tmpdir = os.getenv('TMPDIR', tempfile.gettempdir())
mpl_cache = os.path.join(tmpdir, 'matplotlib-cache')
os.makedirs(mpl_cache, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_cache
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import EsmForMaskedLM, EsmTokenizer

def parse_hgvs(protein_change):
    """Extract WT AA, position, and mutant AA from HGVS string like 'p.A586V'."""
    pattern = r"^p\.([A-Z])(\d+)(?!Ter)([A-Z])"
    m = re.match(pattern, protein_change)
    if not m:
        return None
    wt, pos, mt = m.groups()
    return wt, int(pos), mt

def generate_heatmap(protein_sequence, model, tokenizer, start_pos=1, end_pos=None):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    decoded = tokenizer(protein_sequence, return_tensors="pt").to(device)
    input_ids = decoded['input_ids']
    sequence_length = input_ids.shape[1] - 2

    if end_pos is None:
        end_pos = sequence_length

    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    heatmap = np.zeros((20, end_pos - start_pos + 1))

    for position in range(start_pos, end_pos + 1):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, position] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked_input_ids).logits
            probabilities = torch.nn.functional.softmax(logits[0, position], dim=0)
            log_probabilities = torch.log(probabilities)

        wt_residue = input_ids[0, position].item()
        log_prob_wt = log_probabilities[wt_residue].item()

        for i, aa in enumerate(amino_acids):
            aa_id = tokenizer.convert_tokens_to_ids(aa)
            log_prob_mt = log_probabilities[aa_id].item()
            heatmap[i, position - start_pos] = log_prob_mt - log_prob_wt
            #if (log_prob_mt - log_prob_wt) > 2:
            #   print(f'position: {position}, wt: {wt_residue} with {log_prob_wt}, and mt: {aa_id} with {log_prob_mt}\n')

    return heatmap, amino_acids

def plot_heatmap_with_dots(data, descr, gene, title, sequence, amino_acids, mutation_list, start_pos=1):

    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto")
    plt.xticks(range(len(sequence)), list(sequence))
    plt.yticks(range(len(amino_acids)), amino_acids)
    plt.xlabel("Position in Protein Sequence")
    plt.ylabel("Amino Acid Mutations")
    plt.title(f"{title} 650M")
    plt.colorbar(label="LLR Difference (Mutant − Base)")

    # Overlay mutation markers
    marks_x, marks_y = [], []
    for mut in mutation_list:
        parsed = parse_hgvs(mut)
        if parsed:
            _, pos, mt = parsed
            if mt in amino_acids:
                row = amino_acids.index(mt)
                col = pos - start_pos
                if 0 <= col < data.shape[1]:
                    marks_x.append(col)
                    marks_y.append(row)

    if marks_x:
        plt.scatter(marks_x, marks_y, marker='o', color='black', s=50, label='Mutations')
        plt.legend(loc='upper right')

    plt.tight_layout()

    # Define the path
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run2/heatmaps/{gene}/{title.replace(' ', '_')}.png"
    folder = os.path.dirname(save_path)

    os.makedirs(folder, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_heatmap(descr, params, gene, data, title, sequence, amino_acids):
    plt.figure(figsize=(20, 5))
    plt.imshow(data, cmap="bwr_r" if "Difference" in title else "viridis_r", aspect="auto", vmin=None, vmax=None)
    plt.xticks(range(len(sequence)), list(sequence))
    plt.yticks(range(20), amino_acids)
    plt.xlabel("Position in Protein Sequence")
    plt.ylabel("Amino Acid Mutations")
    plt.title(title + ' ' + str(params) + 'M')
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.tight_layout()
    
    # Define the path
    save_path = f"/g/data/gi52/jaime/trained/esm2_650M_model/{descr}/run2/heatmaps/{gene}/{title.replace(' ', '_')}.png"
    folder = os.path.dirname(save_path)

    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300)

def topk_predictions(model, tokenizer, protein_seq, masked_pos, k=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokens = list(protein_seq)
    tokens[masked_pos - 1] = tokenizer.mask_token  # Replace 1-based index with mask
    masked_seq = "".join(tokens)

    inputs = tokenizer(masked_seq, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        prediction_logits = logits[mask_index]

    probs = torch.softmax(prediction_logits, dim=-1)
    topk = torch.topk(probs, k=k, dim=-1)
    top_tokens = tokenizer.convert_ids_to_tokens(topk.indices[0].tolist())
    top_probs = topk.values[0].tolist()

    return list(zip(top_tokens, top_probs))

def main():

    # Model Parameters (in millions) of finetuned model
    params = 650
    ms_descr = "missense"
    fs_descr = "frameshift"

    # Sequence and gene of interest 
    ''' ms_gene = "OBSCN_1-1022"
    ms_sequence = "MDQPQFSGAPRFLTRPKAFVVSVGKDATLSCQIVGNPTPQVSWEKDQQPVAAGARFRLAQDGDLYRLTILDLALGDSGQYVCRARNAIGEAFAAVGLQVDAEAACAEQAPHFLLRPTSIRVREGSEATFRCRVGGSPRPAVSWSKDGRRLGEPDGPRVRVEELGEASALRIRAARPRDGGTYEVRAENPLGAASAAAALVVDSDAADTASRPGTSTAALLAHLQRRREAMRAEGAPASPPSTGTRTCTVTEGKHARLSCYVTGEPKPETVWKKDGQLVTEGRRHVVYEDAQENFVLKILFCKQSDRGLYTCTASNLVGQTYSSVLVVVREPAVPFKKRLQDLEVREKESATFLCEVPQPSTEAAWFKEETRLWASAKYGIEEEGTERRLTVRNVSADDDAVYICETPEGSRTVAELAVQGNLLRKLPRKTAVRVGDTAMFCVELAVPVGPVHWLRNQEEVVAGGRVAISAEGTRHTLTISQCCLEDVGQVAFMAGDCQTSTQFCVSAPRKPPLQPPVDPVVKARMESSVILSWSPPPHGERPVTIDGYLVEKKKLGTYTWIRCHEAEWVATPELTVADVAEEGNFQFRVSALNSFGQSPYLEFPGTVHLAPKLAVRTPLKAVQAVEGGEVTFSVDLTVASAGEWFLDGQALKASSVYEIHCDRTRHTLTIREVPASLHGAQLKFVANGIESSIRMEVRAAPGLTANKPPAAAAREVLARLHEEAQLLAELSDQAAAVTWLKDGRTLSPGPKYEVQASAGRRVLLVRDVARDDAGLYECVSRGGRIAYQLSVQGLARFLHKDMAGSCVDAVAGGPAQFECETSEAHVHVHWYKDGMELGHSGERFLQEDVGTRHRLVAATVTRQDEGTYSCRVGEDSVDFRLRVSEPKVVFAKEQLARRKLQAEAGASATLSCEVAQAQTEVTWYKDGKKLSSSSKVCMEATGCTRRLVVQQAGQADAGEYSCEAGGQRLSFHLDVKEPKVVFAKDQVAHSEVQAEAGASATLSCEVAQAQTEVMWYKDGKKL"
    '''
    ms_gene = "OBSCN_6360-7382"
    ms_sequence = "QVTIEDVQAQTGGTAQFEAIIEGDPQPSVTWYKDSVQLVDSTRLSQQQEGTTYSLVLRHVASKDAGVYTCLAQNTGGQVLCKAELLVLGGDNEPDSEKQSHRRKLHSFYEVKEEIGRGVFGFVKRVQHKGNKILCAAKFIPLRSRTRAQAYRERDILAALSHPLVTGLLDQFETRKTLILILELCSSEELLDRLYRKGVVTEAEVKVYIQQLVEGLHYLHSHGVLHLDIKPSNILMVHPAREDIKICDFGFAQNITPAELQFSQYGSPEFVSPEIIQQNPVSEASDIWAMGVISYLSLTCSSPFAGESDRATLLNVLEGRVSWSSPMAAHLSEDAKDFIKATLQRAPQARPSAAQCLSHPWFLKSMPAEEAHFINTKQLKFLLARSRWQRSLMSYKSILVMRSIPELLRGPPDSPSLGVARHLCRDTGGSSSSSSSSDNELAPFARAKSLPPSPVTHSPLLHPRGFLRPSASLPEEAEASERSTEAPAPPASPEGAGPPAAQGCVPRHSVIRSLFYHQAGESPEHGALAPGSRRHPARRRHLLKGGYIAGALPGLREPLMEHRVLEEEAAREEQATLLAKAPSFETALRLPASGTHLAPGHSHSLEHDSPSTPRPSSEACGEAQRLPSAPSGGAPIRDMGHPQGSKQLPSTGGHPGTAQPERPSPDSPWGQPAPFCHPKQGSAPQEGCSPHPAVAPCPPGSFPPGSCKEAPLVPSSPFLGQPQAPPAPAKASPPLDSKMGPGDISLPGRPKPGPCSSPGSASQASSSQVSSLRVGSSQVGTEPGPSLDAEGWTQEAEDLSDSTPTLQRPQEQATMRKFSLGGRGGYAGVAGYGTFAFGGDAGGMLGQGPMWARIAWAVSQSEEEEQEEARAESQSEEQQEARAESPLPQVSARPVPEVGRAPTRSSPEPTPWEDIGQVSLVQIRDLSGDAEAADTISLDISEVDPAYLNLSDLYDIKYLPFEFMIFRKVPKSAQPEPPSPMAEEELAEFPEPTWPWPGELGPHAGLEITEESEDVDALLAEAA"
    ms_start_pos = 6360

    fs_gene = "OBSCN_6360-7382"
    fs_sequence = "QVTIEDVQAQTGGTAQFEAIIEGDPQPSVTWYKDSVQLVDSTRLSQQQEGTTYSLVLRHVASKDAGVYTCLAQNTGGQVLCKAELLVLGGDNEPDSEKQSHRRKLHSFYEVKEEIGRGVFGFVKRVQHKGNKILCAAKFIPLRSRTRAQAYRERDILAALSHPLVTGLLDQFETRKTLILILELCSSEELLDRLYRKGVVTEAEVKVYIQQLVEGLHYLHSHGVLHLDIKPSNILMVHPAREDIKICDFGFAQNITPAELQFSQYGSPEFVSPEIIQQNPVSEASDIWAMGVISYLSLTCSSPFAGESDRATLLNVLEGRVSWSSPMAAHLSEDAKDFIKATLQRAPQARPSAAQCLSHPWFLKSMPAEEAHFINTKQLKFLLARSRWQRSLMSYKSILVMRSIPELLRGPPDSPSLGVARHLCRDTGGSSSSSSSSDNELAPFARAKSLPPSPVTHSPLLHPRGFLRPSASLPEEAEASERSTEAPAPPASPEGAGPPAAQGCVPRHSVIRSLFYHQAGESPEHGALAPGSRRHPARRRHLLKGGYIAGALPGLREPLMEHRVLEEEAAREEQATLLAKAPSFETALRLPASGTHLAPGHSHSLEHDSPSTPRPSSEACGEAQRLPSAPSGGAPIRDMGHPQGSKQLPSTGGHPGTAQPERPSPDSPWGQPAPFCHPKQGSAPQEGCSPHPAVAPCPPGSFPPGSCKEAPLVPSSPFLGQPQAPPAPAKASPPLDSKMGPGDISLPGRPKPGPCSSPGSASQASSSQVSSLRVGSSQVGTEPGPSLDAEGWTQEAEDLSDSTPTLQRPQEQATMRKFSLGGRGGYAGVAGYGTFAFGGDAGGMLGQGPMWARIAWAVSQSEEEEQEEARAESQSEEQQEARAESPLPQVSARPVPEVGRAPTRSSPEPTPWEDIGQVSLVQIRDLSGDAEAADTISLDISEVDPAYLNLSDLYDIKYLPFEFMIFRKVPKSAQPEPPSPMAEEELAEFPEPTWPWPGELGPHAGLEITEESEDVDALLAEAA"
    fs_start_pos = 6360

    fs_gene = "ARID1A"


    print(f'Gene being investigated for both models is {ms_gene} with sequence {ms_sequence}')

    '''
    fs_gene = "ARID1A"
    fs_sequence = "MAAQVAPAAASSLGNPPPPPPSELKKAEQQQREEAGGEAAAAAAAERGEMKAAAGQESEGPAVGPPQPLGKELQDGAESNGGGGGGGAGSGGGPGAEPDLKNSNGNAGPRPALNNNLTEPPGGGGGGSSDGVGAPPHSAAAALPPPAYGFGQPYGRSPSAVAAAAAAVFHQQHGGQQSPGLAALQSGGGGGLEPYAGPQQNSHDHGFPNHQYNSYYPNRSAYPPPAPAYALSSPRGGTPGSGAAAAAGSKPPPSSSASASSSSSSFAQQRFGAMGGGGPSAAGGGTPQPTATPTLNQLLTSPSSARGYQGYPGGDYSGGPQDGGAGKGPADMASQCWGAAAAAAAAAAASGGAQQRSHHAPMSPGSSGGGGQPLARTPQPSSPMDQMGKMRPQPYGGTNPYSQQQGPPSGPQQGHGYPGQPYGSQTPQRYPMTMQGRAQSAMGGLSYTQQIPPYGQQGPSGYGQQGQTPYYNQQSPHPQQQQPPYSQQPPSQTPHAQPSYQQQPQSQPPQLQSSQPPYSQQPSQPPHQQSPAPYPSQQSTTQQHPQSQPPYSQPQAQSPYQQQQPQQPAPSTLSQQAAYPQPQSQQSQQTAYSQQRFPPPQELSQDSFGSQASSAPSMTSSKGGQEDMNLSLQSRPSSLPDLSGSIDDLPMGTEGALSPGVSTSGISSSQGEQSNPAQSPFSPHTSPHLPGIRGPSPSPVGSPASVAQSRSGPLSPAAVPGNQMPPRPPSGQSDSIMHPSMNQSSIAQDRGYMQRNPQMPQYSSPQPGSALSPRQPSGGQIHTGMGSYQQNSMGSYGPQGGQYGPQGGYPRQPNYNALPNANYPSAGMAGGINPMGAGGQMHGQPGIPPYGTLPPGRMSHASMGNRPYGPNMANMPPQVGSGMCPPPGGMNRKTQETAVAMHVAANSIQNRPPGYPNMNQGGMMGTGPPYGQGINSMAGMINPQGPPYSMGGTMANNSAGMAASPEMMGLGDVKLTPATKMNNKADGTPKTESKSKKSSSSTTTNEKITKLYELGGEPERKM"

     
    ms_gene = "RYR2"
    ms_sequence = "MADGGEGEDEIQFLRTDDEVVLQCTATIHKEQQKLCLAAEGFGNRLCFLESTSNSKNVPPDLSICTFVLEQSLSVRALQEMLANTVEKSEGQVDVEKWKFMMKTAQGGGHRTLLYGHAILLRHSYSGMYLCCLSTSRSSTDKLAFDVGLQEDTTGEACWWTIHPASKQRSEGEKVRVGDDLILVSVSSERYLHLSYGNGSLHVDAAFQQTLWSVAPISSGSEAAQGYLIGGDVLRLLHGHMDECLTVPSGEHGEEQRRTVHYEGGAVSVHARSLWRLETLRVAWSGSHIRWGQPFRLRHVTTGKYLSLMEDKNLLLMDKEKADVKSTAFTFRSSKEKLDVGVRKEVDGMGTSEIKYGDSVCYIQHVDTGLWLTYQSVDVKSVRMGSIQRKAIMHHEGHMDDGISLSRSQHEESRTARVIRSTVFLFNRFIRGLDALSKKAKASTVDLPIESVSLSLQDLIGYFHPPDEHLEHEDKQNRLRALKNRQNLFQEEGMINLVLECIDRLHVYSSAAHFADVAGREAGESWKSILNSLYELLAALIRGNRKNCAQFSGSLDWLISRLERLEASSGILEVLHCVLVESPEALNIIKEGHIKSIISLLDKHGRNHKVLDVLCSLCVCHGVAVRSNQHLICDNLLPGRDLLLQTRLVNHVSSMRPNIFLGVSEGSAQYKKWYYELMVDHTEPFVTAEATHLRVGWASTEGYSPYPGGGEEWGGNGVGDDLFSYGFDGLHLWSGCIARTVSSPNQHLLRTDDVISCCLDLSAPSISFRINGQPVQGMFENFNIDGLFFPVVSFSAGIKVRFLLGGRHGEFKFLPPPGYAPCYEAVLPKEKLKVEHSREYKQERTYTRDLLGPTVSLTQAAFTPIPVDTSQIVLPPHLERIREKLAENIHELWVMNKIELGWQYGPVRDDNKRQHPCLVEFSKLPEQERNYNLQMSLETLKTLLALGCHVGISDEHAEDKVKKMKLPKNYQLTSGYKPAPMDLSFIKLTPSQEAMVDKLAENAHNVWARDRIRQGWTYGIQQ"
    fs_gene = "RYR2"
    fs_sequence = "MADGGEGEDEIQFLRTDDEVVLQCTATIHKEQQKLCLAAEGFGNRLCFLESTSNSKNVPPDLSICTFVLEQSLSVRALQEMLANTVEKSEGQVDVEKWKFMMKTAQGGGHRTLLYGHAILLRHSYSGMYLCCLSTSRSSTDKLAFDVGLQEDTTGEACWWTIHPASKQRSEGEKVRVGDDLILVSVSSERYLHLSYGNGSLHVDAAFQQTLWSVAPISSGSEAAQGYLIGGDVLRLLHGHMDECLTVPSGEHGEEQRRTVHYEGGAVSVHARSLWRLETLRVAWSGSHIRWGQPFRLRHVTTGKYLSLMEDKNLLLMDKEKADVKSTAFTFRSSKEKLDVGVRKEVDGMGTSEIKYGDSVCYIQHVDTGLWLTYQSVDVKSVRMGSIQRKAIMHHEGHMDDGISLSRSQHEESRTARVIRSTVFLFNRFIRGLDALSKKAKASTVDLPIESVSLSLQDLIGYFHPPDEHLEHEDKQNRLRALKNRQNLFQEEGMINLVLECIDRLHVYSSAAHFADVAGREAGESWKSILNSLYELLAALIRGNRKNCAQFSGSLDWLISRLERLEASSGILEVLHCVLVESPEALNIIKEGHIKSIISLLDKHGRNHKVLDVLCSLCVCHGVAVRSNQHLICDNLLPGRDLLLQTRLVNHVSSMRPNIFLGVSEGSAQYKKWYYELMVDHTEPFVTAEATHLRVGWASTEGYSPYPGGGEEWGGNGVGDDLFSYGFDGLHLWSGCIARTVSSPNQHLLRTDDVISCCLDLSAPSISFRINGQPVQGMFENFNIDGLFFPVVSFSAGIKVRFLLGGRHGEFKFLPPPGYAPCYEAVLPKEKLKVEHSREYKQERTYTRDLLGPTVSLTQAAFTPIPVDTSQIVLPPHLERIREKLAENIHELWVMNKIELGWQYGPVRDDNKRQHPCLVEFSKLPEQERNYNLQMSLETLKTLLALGCHVGISDEHAEDKVKKMKLPKNYQLTSGYKPAPMDLSFIKLTPSQEAMVDKLAENAHNVWARDRIRQGWTYGIQQ"
    '''
    # Load training samples and split 
    data_path = Path("./data")
    test_size = 0.2
    valid_size = 0.25

    ms_df = pd.read_parquet(data_path / "all_ms_samples.parquet")
    ms_train_df, ms_test_df = train_test_split(ms_df, test_size=test_size, random_state=0)
    ms_train_df, ms_valid_df = train_test_split(ms_train_df, test_size=valid_size, random_state=0)
    
    fs_df = pd.read_parquet(data_path / "all_fs_samples.parquet")
    fs_train_df, fs_test_df = train_test_split(fs_df, test_size=test_size, random_state=0)
    fs_train_df, fs_valid_df = train_test_split(fs_train_df, test_size=valid_size, random_state=0)


    # Filter rows and print relevant columns
    ms_filtered = (
        ms_train_df[ms_train_df['HugoSymbol'] == ms_gene][['ProteinChange']]
    )
    ms_mutation_list = ms_filtered['ProteinChange'].tolist()

    fs_filtered = (
        fs_train_df[fs_train_df['HugoSymbol'] == fs_gene][['ProteinChange']] 
    )
    fs_mutation_list = fs_filtered['ProteinChange'].tolist()


    # changed!!
    fs_filtered = (
    fs_train_df[fs_train_df['HugoSymbol'] == fs_gene][['ProteinChange']]
    .assign(parsed=lambda df: df['ProteinChange'].apply(parse_hgvs)))

    # Split tuple into separate columns
    fs_filtered[['wt', 'pos', 'mt']] = pd.DataFrame(fs_filtered['parsed'].tolist(), index=fs_filtered.index)

    # Drop the tuple column
    fs_filtered = fs_filtered.drop(columns=['parsed'])

    # Sort by pos
    fs_filtered = fs_filtered.sort_values(by='pos', ascending=True).reset_index(drop=True)
    fs_filtered = fs_filtered[(fs_filtered['pos'] >= 450) & (fs_filtered['pos'] <= 1120)]
    

    # Load original ESM-2 model
    base_model_name = f"/g/data/gi52/jaime/esm2_{params}M_model"
    base_tokenizer = EsmTokenizer.from_pretrained(base_model_name)
    base_model = EsmForMaskedLM.from_pretrained(base_model_name)

    # Load missense fine-tuned model
    ms_model_path = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/{ms_descr}/run2/final_merged"
    ms_tokenizer = EsmTokenizer.from_pretrained(ms_model_path)
    ms_model = EsmForMaskedLM.from_pretrained(ms_model_path)

    # Load Frameshift fine-tuned model
    fs_model_path = f"/g/data/gi52/jaime/trained/esm2_{params}M_model/{fs_descr}/run2/final_merged"
    fs_tokenizer = EsmTokenizer.from_pretrained(fs_model_path)
    fs_model = EsmForMaskedLM.from_pretrained(fs_model_path)


    # Generate heatmaps
    ms_base_heatmap, amino_acids = generate_heatmap(ms_sequence, base_model, base_tokenizer)
    ms_heatmap, _ = generate_heatmap(ms_sequence, ms_model, ms_tokenizer)

    fs_base_heatmap, amino_acids = generate_heatmap(fs_sequence, base_model, base_tokenizer)
    fs_heatmap, _ = generate_heatmap(fs_sequence, fs_model, fs_tokenizer)

    # Compute difference
    ms_diff_heatmap = ms_heatmap - ms_base_heatmap
    fs_diff_heatmap = fs_heatmap - fs_base_heatmap

    plot_heatmap("original", params, ms_gene, ms_base_heatmap, "Original ESM2 Model (LLRs)", ms_sequence, amino_acids)
    plot_heatmap("missense", params, ms_gene, ms_heatmap, "Fine-tuned Missense Model (LLRs)", ms_sequence, amino_acids)
    plot_heatmap("missense", params, ms_gene, ms_diff_heatmap, "Difference (Fine-tuned Missense - Original)", ms_sequence, amino_acids)

    plot_heatmap("original", params, fs_gene, fs_base_heatmap, "Original ESM2 Model (LLRs)", fs_sequence, amino_acids)
    plot_heatmap("frameshift", params, fs_gene, fs_heatmap, "Fine-tuned Frameshift Model (LLRs)", fs_sequence, amino_acids)
    plot_heatmap("frameshift", params, fs_gene, fs_diff_heatmap, "Difference (Fine-tuned Frameshift - Original)", fs_sequence, amino_acids)

    plot_heatmap_with_dots(ms_diff_heatmap, "missense", ms_gene, "Difference (Fine-tuned Missense - Original) with Mutations", ms_sequence, amino_acids, ms_mutation_list, start_pos=ms_start_pos)
    plot_heatmap_with_dots(fs_diff_heatmap, "frameshift", fs_gene, "Difference (Fine-tuned Frameshift - Original) with Mutations", fs_sequence, amino_acids, fs_mutation_list, start_pos=fs_start_pos)

    # Compare amino acid predictions
    ms_masked_pos = 6530
    fs_masked_pos = 6530

    ms_original_preds = topk_predictions(base_model, base_tokenizer, ms_sequence, ms_masked_pos)
    ms_preds = topk_predictions(ms_model, ms_tokenizer, ms_sequence, ms_masked_pos)

    fs_original_preds = topk_predictions(base_model, base_tokenizer, fs_sequence, fs_masked_pos)
    fs_preds = topk_predictions(fs_model, fs_tokenizer, fs_sequence, fs_masked_pos)

    print(f"Original model top predictions for gene {ms_gene} at position {ms_masked_pos}:", ms_original_preds)
    print(f"Fine-tuned missense model top predictions for gene {ms_gene} at position {ms_masked_pos}:", ms_preds)
    print('\n')
    print(f"Original model top predictions for gene {fs_gene} at position {fs_masked_pos}:", fs_original_preds)
    print(f"Fine-tuned frameshift model top predictions for gene {fs_gene} at position {fs_masked_pos}:", fs_preds)


if __name__ == '__main__':
    main()