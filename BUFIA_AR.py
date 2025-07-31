
# -*- coding: utf-8 -*-

from queue import Queue
import argparse
from autorep import Autorep
import os


def convert_to_ar(filename, output_dir = 'output'):
    """
    Convert a text file to a list of distinct Autorep objects.
    """
    print("Converting words to ARs...")
    autorep_list = []
    try:
        with open(filename, "r") as file:
            for line in file:
                asr = Autorep(line.strip())
                autorep_list.append(asr)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return []

    autoset = set()
    unique_autoreps = []

    for i in autorep_list:
        info = i.info()
        if info not in autoset:
            # pprint.pprint(i)
            unique_autoreps.append(i)
            autoset.add(info)
        # else:
        #     print(f"Duplicate found: {i.word}")

    
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'ar_list.txt')

        
    try:
        with open(file_path, "w") as file:
            for ar in unique_autoreps:
                file.write(f"{ar.word}, {ar.ocp_mel}, {ar.assoc} \n")  # or whatever field you want
        print(f'Processed {len(autorep_list)} words, found {len(unique_autoreps)} unique ASRs')
    
    except Exception as e:
        print(f"Failed to write output: {e}")

    return unique_autoreps




def check_from_data(ar, positive_data, find=False):
    """Check if an Autorep object `ar` is a substructure of any in `positive_data`."""

    for i in positive_data:
        if i.check_adj_contain(ar):
            if find:
                print(i.word,i)
            return True
    return False


def not_contain_forbidden_piece(ar, grammar):
    
    """Check that `ar` does NOT contain any structure from the `grammar` set."""
    for i in grammar:
        # print(f"checking {ar} against {i}")
        if ar.check_adj_contain(i):
            return False
    return True

def bufia(D, t=2, s=2, m=2, format='svg', output_dir='output'):
    """
    Breadth-first grammar induction algorithm from positive data D,
    with complexity thresholds t, s, m.
    """
    t_threshold = t
    s_threshold = s
    m_threshold = m

    G = set()
    Q = Queue()
    V = set()
    s0 = Autorep()  # Empty starting structure
    Q.put(s0)

    while not Q.empty():
        s = Q.get()
        V.add(s)

        if check_from_data(s, D):  # Structure fits data
            S = s.next_ar(3)
            for i in S:
                try:
                    if (
                        i in V
                        or not not_contain_forbidden_piece(i, G)
                        or i.get_max('t') > t_threshold
                        or i.get_max('s') > s_threshold
                        or i.get_max('m') > m_threshold
                    ):
                        continue
                    Q.put(i)
                except IndexError as e:
                    print(f"IndexError in candidate {i}: {e}")
                    continue
        else:
            if s not in G and not_contain_forbidden_piece(s, G):
                G.add(s)

    # Create output dir once
    os.makedirs(output_dir, exist_ok=True)

    # Draw each grammar rule found
    for rule in G:
        rule.draw(suffix=format, output_dir=output_dir)

    print(f'Found {len(G)} constraints')
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BUFIA-AR")
    parser.add_argument('--input', type=str, required=True, help='input syllabified wordlist')
    parser.add_argument('--output', type=str, default='output', help='filepath to save the learned graphs')
    parser.add_argument('--format', type=str, default='svg', help='save plots in what format (e.g., svg, png, pdf)')
    parser.add_argument('--t', type=int, default=2, help='tone number limit')
    parser.add_argument('--s', type=int, default=2, help='syllable number limit')
    parser.add_argument('--m', type=int, default=2, help='mora number limit')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-f', '--file', action='store_true', help="Print the output of the autolist (all Autorep objects)")
    group.add_argument('-learn', action='store_true', help="Learn and print a list of grammar rules")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file '{args.input}' does not exist!")
        exit(1)

    autolist = convert_to_ar(args.input, output_dir= args.output)

    if args.file:
        print("=== Autolist ===")
        for item in autolist:
            print(item)

    if args.learn:
        grammar = bufia(autolist, t=args.t, s=args.s, m=args.m, format=args.format, output_dir=args.output)
        print("=== Learned grammar ===")
        print(f"{len(grammar)} was found")
