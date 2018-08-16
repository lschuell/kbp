import random
import logging
from collections import defaultdict as dd
import re

logger = logging.getLogger(__name__)

def __get_next_pred_obj(sub_pred_obj_dict, current_resource):
    pred_obj_map = sub_pred_obj_dict.get(current_resource)
    if pred_obj_map is None:
        return (None, None)
    choosen_predicate = random.choice(list(pred_obj_map.keys()))
    objects = pred_obj_map[choosen_predicate]
    choosen_object = next(iter(random.sample(objects, 1)))
    return (choosen_predicate, choosen_object)

def write_random_walks(sub_pred_obj_dict, out_path, number_of_walks_per_resource = 20, maximum_length_of_a_walk = 50, only_unique_walks = True):

    with open(out_path, 'w', encoding='utf-8') as out_file:
        subject_count = len(sub_pred_obj_dict.keys())
        for i, resource in enumerate(sub_pred_obj_dict.keys()):
            if only_unique_walks:
                unique_walks = set()
            else:
                unique_walks = []
            #unique_walks = [] if only_unique_walks else set()
            for k in range(number_of_walks_per_resource):
                current = resource
                one_walk = [current]
                for l in range(maximum_length_of_a_walk):
                    (choosen_predicate, choosen_object) = __get_next_pred_obj(sub_pred_obj_dict, current)
                    if choosen_predicate is None or choosen_object is None:
                        break
                    one_walk.append(choosen_predicate)
                    one_walk.append(choosen_object)
                    current = choosen_object
                if only_unique_walks:
                    unique_walks.add(tuple(one_walk))
                else:
                    unique_walks.append(tuple(one_walk))

            for walk in unique_walks:
                out_file.write(" ".join(walk) + '\n')

            #if i > 3:
            #    break
            #print(i)
            if i % 1000 == 0:
                logger.info("%d / %d", i, subject_count)

def get_sub_pred_obj_dict(KG="darkscape"):
    sub_pred_obj_dict = dd(lambda: dd(set))
    with open("KGs_for_gold_standard/"+KG+"/enwiki-20170801-infobox-properties-redirected.ttl",'r') as src_file:
        for i, line in enumerate(src_file):
            starts = [m.start() for m in re.finditer('<', line)]
            ends = [m.start() for m in re.finditer('>', line)]
            if (len(starts) == 3) & (len(ends) == 3):
                check = "http://dbkwik.webdatacommons.org/" + KG
                sub = line[starts[0]:ends[0]+1]
                if check not in sub:
                    continue
                pred = line[starts[1]:ends[1]+1]
                if check not in pred:
                    continue
                obj = line[starts[2]:ends[2]+1]
                if check not in obj:
                    continue

                sub_pred_obj_dict[sub][pred].add(obj)

    return sub_pred_obj_dict

if __name__ == "__main__":
    KG = "oldschoolrunescape"
    sub_pred_obj_dict = get_sub_pred_obj_dict(KG=KG)
    write_random_walks(sub_pred_obj_dict, "./rwalks/rwalk_"+KG+".txt", number_of_walks_per_resource = 1000, maximum_length_of_a_walk = 500, only_unique_walks = True)