#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 08:25:59 2019

ldsc simulation framework

@author: nbaya
"""

import hail as hl
from hail.expr.expressions import expr_int32, expr_int64, expr_float32, expr_float64
from hail.typecheck import typecheck, oneof, nullable
from hail.matrixtable import MatrixTable
import re
from datetime import datetime, timedelta

@typecheck(mt=MatrixTable, 
           genotype=oneof(expr_int32,
                          expr_int64, 
                          expr_float32, 
                          expr_float64),
           h2=oneof(nullable(float),
                    nullable(int)),
           pi=oneof(float,int),
           is_annot_inf=bool,
           annot_coef_dict=nullable(dict),
           annot_regex=nullable(str),
           h2_normalize=bool,
           is_popstrat=bool,
           cov_coef_dict=nullable(dict),
           cov_regex=nullable(str),
           path_to_save=nullable(str))
def simulate_phenotypes(mt, genotype, h2=None, pi=1, is_annot_inf=False, annot_coef_dict=None,
                        annot_regex=None,h2_normalize=True, is_popstrat=False, cov_coef_dict=None,
                        cov_regex=None, path_to_save=None):
    r'''Simulate phenotypes for testing LD score regression.
    
    Simulates betas (SNP effects) under the infinitesimal, spike & slab, or 
    annotation-informed models, depending on parameters passed. Optionally adds
    population stratification.
    
    Parameters
    ----------
    mt : :class:`.MatrixTable`
        MatrixTable containing genotypes to be used. Also should contain 
        variant annotations as row fields if running the annotation-informed
        model or covariates as column fields if adding population stratification.
    genotype : :class:`.Expression`
        Entry field containing genotypes of individuals to be used for the
        simulation.
    h2 : :obj:`float` or :obj:`int`
        Heritability of simulted trait. Can only be None if running annotation-
        informed model.
    pi : :obj:`float` or :obj:`int`
        Probability of SNP being causal when simulating under the spike & slab 
        model.
    is_annot_inf : :obj:`bool`
        Whether to simulate under the annotation-informed model. 
        Requires annot_coef_dict and annot_regex to not both be None.
    annot_coef_dict : :obj:`dict` from :obj:`str` to :obj:`float`
        Dictionary with annotation row field names as keys and coefficients for
        each annotation as values. Coefficients are equivalent to tau values in 
        partitioned heritability.
    annot_regex : :obj:`str`
        Regex to search for annotations to use in an annotation-informed model.
    h2_normalize : :obj:`bool`
        Whether to normalize h2 when running an annotation-informed model.
        Requires is_annot_inf=True and h2!=None.
    is_popstrat : :obj:`bool`
        Whether to simulate with population stratification. 
        Requires cov_coef_dict and cov_regex to not both be None.
    cov_coef_dict : :obj:`dict` from :obj:`str` to :obj:`float`
        Dictionary with covariate column field names as keys and coefficients 
        for each covariate as values.
    cov_regex : :obj:`str`
        Regex to search for covariates to add population stratification.
    path_to_save : :obj:`str`
        Path to save MatrixTable of simulation results.
    
    Returns
    -------
    :class:`.MatrixTable`
        MatrixTable with simulated betas and phenotypes, simulated according
        to user-specified model.
    '''
    check_beta_args(h2=h2,pi=pi,is_annot_inf=is_annot_inf,annot_coef_dict=annot_coef_dict,
                        annot_regex=annot_regex,h2_normalize=h2_normalize)
    check_popstrat_args(is_popstrat=is_popstrat,cov_coef_dict=cov_coef_dict,cov_regex=cov_regex)
    starttime = datetime.now()
    print_header(h2=h2, 
                 pi=pi, 
                 is_annot_inf=is_annot_inf, 
                 h2_normalize=h2_normalize,
                 is_popstrat=is_popstrat, 
                 path_to_save=path_to_save)
    mt1 = annotate_w_temp_fields(mt=mt, 
                                 genotype=genotype,
                                 h2=h2, 
                                 pi=pi, 
                                 is_annot_inf=is_annot_inf,
                                 annot_coef_dict=annot_coef_dict,
                                 annot_regex=annot_regex,
                                 is_popstrat=is_popstrat, 
                                 cov_coef_dict=cov_coef_dict,
                                 cov_regex=cov_regex,)
    mt2 = make_betas(mt=mt1, 
                     h2=h2, 
                     pi=pi, 
                     is_annot_inf=is_annot_inf,
                     annot_coef_dict=annot_coef_dict,
                     annot_regex=annot_regex,
                     h2_normalize=h2_normalize)   
    mt2 = mt2.rename({'__beta':'__beta_temp'})
    mt3 =  calculate_phenotypes(mt=mt2, 
                                genotype=mt2.__gt_temp, 
                                h2=h2, 
                                beta=mt2.__beta_temp,
                                is_popstrat=is_popstrat,
                                cov_coef_dict=cov_coef_dict,
                                cov_regex=cov_regex)
    mt4 = clean_fields(mt3, '_temp')
    stoptime = datetime.now()
    runtime = stoptime-starttime
    mt5 = add_sim_description(mt=mt4,h2=h2,starttime=starttime,stoptime=stoptime,
                              runtime=runtime,pi=pi,is_annot_inf=is_annot_inf,
                              annot_coef_dict=annot_coef_dict,annot_regex=annot_regex,
                              h2_normalize=h2_normalize,is_popstrat=is_popstrat,
                              cov_coef_dict=cov_coef_dict,cov_regex=cov_regex,
                              path_to_save=path_to_save)
    print('\rFinished simulation! (runtime={} min)'.format(round((runtime.total_seconds())/60, 4)).ljust(100))
    if path_to_save is not None:
        print(f'\rWriting simulation to: {path_to_save}')
        mt5 = mt5.checkpoint(path_to_save)
    return mt5

@typecheck(h2=oneof(nullable(float),
                    nullable(int)),
           pi=oneof(float,int),
           is_annot_inf=bool,
           annot_coef_dict=nullable(dict),
           annot_regex=nullable(str),
           h2_normalize=bool)
def check_beta_args(h2=None, pi=1, is_annot_inf=False, annot_coef_dict=None, 
                    annot_regex=None, h2_normalize=True):
    '''checks beta args for simulate() and make_betas()'''
    if is_annot_inf: #if using the annotation-informed model
        assert (annot_coef_dict != None or annot_regex != None), 'If using annotation-informed model, annot_coef_dict and annot_regex cannot both be None'
        if h2_normalize:
            assert (h2 != None), 'h2 cannot be None when h2_normalize=True'
            assert (h2 >= 0 and h2 <= 1), 'h2 must be in [0,1]'
        if not h2_normalize and h2 != None and not (h2 >= 0 and h2 <= 0):
            print('>> Ignoring non-valid h2={} (not in [0,1]) because h2_normalize=False'.format(h2))
    else:
        assert (h2 != None), 'h2 cannot be None, unless running annotation-informed model'
        assert (h2 >= 0 and h2 <= 1), 'h2 must be in [0,1]'
        assert (pi >= 0 and pi <= 1), 'pi must be in [0,1]'
        assert h2_normalize == True, 'h2_normalize cannot be true unless running annotation-informed model'
        if annot_coef_dict != None or annot_regex != None:
            print('>> Ignoring annotation-informed-related args because is_annot_inf=False')

@typecheck(is_popstrat=bool,
           cov_coef_dict=nullable(dict),
           cov_regex=nullable(str))
def check_popstrat_args(is_popstrat=True, cov_coef_dict=None, cov_regex=None):
    '''checks popstrat args for simulate() and add_popstrat()'''
    if is_popstrat:
        assert cov_coef_dict != None or cov_regex != None, 'If adding population stratification, cov_coef_dict and cov_regex cannot both be None'
    else:
        if cov_coef_dict != None or cov_regex != None:
            print('>> Ignoring population stratification-related args because is_popstrat=False')

@typecheck(h2=oneof(nullable(float),
                    nullable(int)),
           pi=oneof(float,int),
           is_annot_inf=bool,
           h2_normalize=bool,
           is_popstrat=bool,
           path_to_save=nullable(str))
def print_header(h2, pi, is_annot_inf, h2_normalize, is_popstrat, path_to_save):
    '''Makes the header for the simulation'''
    header =  '\r****************************************\n'
    header += 'Running simulation framework\n'
    header += 'h2 = {}\n'.format(h2) if (h2_normalize and h2 != None) else ''
    header += 'pi = {} (default: 1)\n'.format(pi)
    header += 'Annotation-informed betas?: {}\n'.format('YES' if is_annot_inf else 'NO')
    header += 'h2-normalized betas?: {}\n'.format('YES' if h2_normalize else 'NO')
    header += 'Add population stratification?: {}\n'.format('YES' if is_popstrat else 'NO')    
    header += '' if path_to_save is None else 'Saving to: {}\n'.format(path_to_save)
    header += '****************************************'
    print(header)

@typecheck(mt=MatrixTable, 
           genotype=oneof(expr_int32,
                          expr_int64, 
                          expr_float32, 
                          expr_float64),
           h2=oneof(nullable(float),
                    nullable(int)),
           pi=oneof(float,int),
           is_annot_inf=bool,
           annot_coef_dict=nullable(dict),
           annot_regex=nullable(str),
           h2_normalize=bool,
           is_popstrat=bool,
           cov_coef_dict=nullable(dict),
           cov_regex=nullable(str))
def annotate_w_temp_fields(mt, genotype, h2=None, pi=1, is_annot_inf=False, 
                           annot_coef_dict=None,annot_regex=None, h2_normalize=True,
                           is_popstrat=False, cov_coef_dict=None, cov_regex=None):
    '''Annotate mt with temporary fields of simulation parameters'''
    check_mt_sources(mt=mt,genotype=genotype)
    return mt._annotate_all(entry_exprs={'__gt_temp':genotype},
                           global_exprs={'__h2_temp':none_to_null(h2), 
                                         '__pi_temp':pi,
                                         '__is_annot_inf_temp':is_annot_inf,
                                         '__annot_coef_dict_temp':none_to_null(annot_coef_dict),
                                         '__annot_regex_temp':none_to_null(annot_regex),
                                         '__h2_normalize_temp':h2_normalize,
                                         '__is_popstrat_temp':is_popstrat,
                                         '__cov_coef_dict_temp':none_to_null(cov_coef_dict),
                                         '__cov_regex_temp':none_to_null(cov_regex)})

@typecheck(mt=MatrixTable,
           genotype=oneof(expr_int32,
                          expr_float64),
           beta=nullable(expr_float64))
def check_mt_sources(mt,genotype,beta=None):
    '''checks that mt matches source mt of genotype and popstrat'''
    if beta is not None:
        assert(mt == genotype._indices.source and mt == beta._indices.source), 'mt must match mt source of genotype and beta'
    else:
        assert(mt == genotype._indices.source), 'mt must match mt source of genotype'
    
@typecheck(arg=oneof(nullable(float),
                     nullable(int),
                     nullable(dict),
                     nullable(str),
                     nullable(expr_int32),
                     nullable(expr_float64)))
def none_to_null(arg):
    '''Converts arg to hl null representation if arg is None'''
    if arg is None:
        return hl.null('str')
    else:
        return arg

@typecheck(mt=MatrixTable, 
           h2=oneof(nullable(float),
                    nullable(int)),
           pi=oneof(float,int),
           is_annot_inf=bool,
           annot_coef_dict=nullable(dict),
           annot_regex=nullable(str),
           h2_normalize=bool)
def make_betas(mt, h2=None, pi=1, is_annot_inf=False, annot_coef_dict=None, annot_regex=None, h2_normalize=True):
    '''Simulate betas. Options: Infinitesimal model, spike & slab, annotation-informed'''  
    check_beta_args(h2=h2,pi=pi,is_annot_inf=is_annot_inf,annot_coef_dict=annot_coef_dict,
                    annot_regex=annot_regex,h2_normalize=h2_normalize)
    M = mt.count_rows()
    if is_annot_inf:
        print('\rSimulating {} annotation-informed betas {}'.format(
                'h2-normalized' if h2_normalize else '',
                '(default coef: 1)' if annot_coef_dict is None else 'using annot_coef_dict'))
        mt1 = agg_fields(mt=mt,coef_dict=annot_coef_dict,regex=annot_regex)
        annot_sum = mt1.aggregate_rows(hl.agg.sum(mt1.__agg_annot))
        return mt1.annotate_rows(__beta = hl.rand_norm(0, hl.sqrt(mt1.__agg_annot*(h2/annot_sum if h2_normalize else 1)))) # if is_h2_normalized: scale variance of betas to be h2, else: keep unscaled variance
    else:
        print('Simulating betas using {} model w/ h2 = {}'.format(('infinitesimal' if pi is 1 else 'spike & slab'),h2))
        mt1 = mt.annotate_globals(__h2 = none_to_null(h2), __pi = pi)
        return mt1.annotate_rows(__beta = hl.rand_bool(pi)*hl.rand_norm(0,hl.sqrt(h2/(M*pi))))

@typecheck(mt=MatrixTable,
           coef_dict=nullable(dict),
           regex=nullable(str),
           axis=str)
def agg_fields(mt,coef_dict=None,regex=None,axis='rows'):
    '''Aggregates fields by linear combination. The coefficient are specified
    by coef_dict value, the row (or col) field name is specified by coef_dict key.
    By default, it searches row field annotations.'''
    assert (regex != None or coef_dict != None), "regex and coef_dict cannot both be None"
    assert axis is 'rows' or axis is 'cols', "axis must be 'rows' or 'cols'"
    coef_dict = get_coef_dict(mt=mt,regex=regex, coef_ref_dict=coef_dict,axis=axis)
    if axis == 'rows':
        mt = mt.annotate_rows(__agg_annot = 0)
        mt = mt.annotate_globals(__annot_coef_dict = none_to_null(coef_dict),
                                 __annot_regex = none_to_null(regex))
    elif axis == 'cols':
        mt = mt.annotate_cols(__agg_cov = 0)
        mt = mt.annotate_globals(__cov_coef_dict = none_to_null(coef_dict),
                                 __cov_regex = none_to_null(regex))
    axis_field = 'annot' if axis=='rows' else 'cov'
    print(f'Fields and associated coefficients used in {axis_field} aggregation: {coef_dict}')
    for field,coef in coef_dict.items():
        if axis == 'rows':
            mt = mt.annotate_rows(__agg_annot = mt.__agg_annot+coef*mt[field])
        elif axis == 'cols':
            mt = mt.annotate_cols(__agg_cov = mt.__agg_cov+coef*mt[field])
    return mt

@typecheck(mt=MatrixTable,
           regex=nullable(str),
           coef_ref_dict=nullable(dict),
           axis=str)
def get_coef_dict(mt, regex=None, coef_ref_dict=None,axis='rows'):
    '''Gets annotations matching regex and pairs with coefficient reference dict
    Number of annotations returned by annotation search should be less than or 
    equal to number of keys in coef_ref_dict. By default, this searches row field
    annotations.'''
    assert (regex != None or coef_ref_dict != None), "regex and coef_ref_dict cannot both be None"
    assert axis is 'rows' or axis is 'cols', "axis must be 'rows' or 'cols'"
    fields_to_search = (mt.row if axis=='rows' else mt.col)
    axis_field = 'annotation' if axis=='rows' else 'covariate' #when axis='rows' we're searching for annotations, axis='cols' searching for covariates
    if regex is None: 
        coef_dict = {k: coef_ref_dict[k] for k in coef_ref_dict.keys() if k in fields_to_search} # take all row (or col) fields in mt matching keys in coef_dict
        assert len(coef_dict) > 0, f'None of the keys in coef_ref_dict match any {axis[:-1]} fields' #if intersect is empty: return error
        return coef_dict #return subset of coef_ref_dict
    else:
        pattern = re.compile(regex)
        fields = [rf for rf in list(fields_to_search) if pattern.match(rf)] #regex search in list of row (or col) fields
        assert len(fields) > 0, f'No {axis[:-1]} fields matched regex search: {regex}'
        if coef_ref_dict is None:
            print(f'Assuming coef = 1 for all {axis_field}s')
            return {k: 1 for k in fields}
        in_coef_ref_dict = set(fields).intersection(set(coef_ref_dict.keys())) #fields in coef_ref_dict
        if in_coef_ref_dict != set(fields): # if >0 fields returned by search are not in coef_ref_dict
            assert len(in_coef_ref_dict) > 0, f'None of the {axis_field} fields in coef_ref_dict match search results' # if none of the fields returned by search are in coef_ref_dict
            fields_to_ignore=set(fields).difference(in_coef_ref_dict)
            print(f'Ignored fields from {axis_field} search: {fields_to_ignore}')
            print('To include ignored fields, change regex to match desired fields')
            fields = list(in_coef_ref_dict)
        return {k: coef_ref_dict[k] for k in fields}

def make_tau_ref_dict():
    '''Make tau_ref_dict from tsv?/dataframe?/Hail Table?'''
    pass

@typecheck(mt=MatrixTable,
           field_list=oneof(list,
                            dict),
           regex_pattern=str)
def add_regex_pattern(mt, field_list, regex_pattern, prefix=True, axis='rows'):
    '''Adds a given pattern to the names of row (or col) field annotations listed in a 
    list or from the keys of a dict. Helpful for searching for those annotations 
    in get_annot_coef_dict() or agg_annotations(). If prefix=False, the pattern will be
    added as a suffix.'''
    if type(field_list) == dict :
        field_list = list(field_list.keys)
        
    axis_fields = set(mt.row) if axis=='rows' else set(mt.col)
    if set(field_list) != axis_fields:
        field_intersection= set(field_list).intersection(axis_fields) #set of fields in both field_list and the mt row (or col) fields
        if len(field_intersection) == 0:
            print(f'No {axis[:-1]} fields matched fields in field_list')
        else:
            field_difference=set(field_list).difference(field_intersection)
            print(f'Ignored {axis[:-1]} fields for adding regex pattern: {field_difference}')
    for field in field_list:
        new_field = [regex_pattern,field][prefix,prefix is True].join('')
        if new_field in axis_fields:
            print(f'{axis[:-1]} field name collision: {new_field}')
            print(f'To avoid name collision, rename field {new_field}')
        else:
            mt = mt._annotate_all(row_exprs={new_field:mt[field]})
    return mt

@typecheck(mt=MatrixTable, 
           genotype=oneof(expr_int32, 
                          expr_int64, 
                          expr_float32, 
                          expr_float64),
           h2=oneof(nullable(float),
                    nullable(int)),
           beta=expr_float64,
           is_popstrat=bool,
           cov_coef_dict=nullable(dict),
           cov_regex=nullable(str))
def calculate_phenotypes(mt, genotype, h2, beta, is_popstrat=False, cov_coef_dict=None,
                         cov_regex=None):
    '''Calculates phenotypes given betas and genotypes. Adding population stratification is optional'''
    check_mt_sources(mt,genotype,beta)
    check_popstrat_args(is_popstrat=is_popstrat,cov_coef_dict=cov_coef_dict,cov_regex=cov_regex)
    mt1 = mt._annotate_all(row_exprs={'__beta':beta},
                           entry_exprs={'__gt':genotype},
                           global_exprs={'__is_popstrat':is_popstrat,
                                         '__cov_coef_dict':none_to_null(cov_coef_dict),
                                         '__cov_regex':none_to_null(cov_regex)})
    mt2 = normalize_genotypes(mt1.__gt)
    print('\rCalculating phenotypes{}...'.format(' w/ population stratification' if is_popstrat else '').ljust(81))
    mt3 = mt2.annotate_cols(__y_no_noise = hl.agg.sum(mt2.__beta * mt2.__norm_gt))
    if h2 is None:
        h2 = mt3.aggregate_cols(hl.agg.stats(mt3.__y_no_noise)).stdev**2
        if h2 > 1:
            print(f'WARNING: Total SNP-based h2 = {h2} (>1)')
            print('Not adding environmental noise')
            h2=1
    mt4 = mt3.annotate_cols(__y = mt3.__y_no_noise + hl.rand_norm(0,hl.sqrt(1-h2)))            
    if is_popstrat:
        return add_popstrat(mt4, 
                             y=mt4.__y, 
                             cov_coef_dict=cov_coef_dict,
                             cov_regex=cov_regex)
    else:
        return mt4
        
@typecheck(genotypes=oneof(expr_int32,
                          expr_int64, 
                          expr_float32, 
                          expr_float64))
def normalize_genotypes(genotypes):
    '''Normalizes genotypes'''
    print('\rNormalizing genotypes...'.ljust(81))
    mt = genotypes._indices.source #get source matrix table of genotypes
    mt1 = mt.annotate_entries(__gt = genotypes)
    mt2 = mt1.annotate_rows(__gt_stats = hl.agg.stats(mt1.__gt))
    return mt2.annotate_entries(__norm_gt = (mt2.__gt-mt2.__gt_stats.mean)/mt2.__gt_stats.stdev)  

@typecheck(mt=MatrixTable, 
           y=oneof(expr_int32,
                   expr_float64),
           cov_coef_dict=nullable(dict),
           cov_regex=nullable(str))
def add_popstrat(mt, y, cov_coef_dict=None, cov_regex=None):
    '''Adds popstrat to a phenotype'''
    check_popstrat_args(cov_coef_dict=cov_coef_dict,cov_regex=cov_regex)
    print('\rAdding population stratification...'.ljust(81))
    mt = mt.annotate_cols(__y = y)
    mt = mt.annotate_globals(__cov_coef_dict=none_to_null(cov_coef_dict),
                             __cov_regex=none_to_null(cov_regex))
    mt1 = agg_fields(mt, coef_dict=cov_coef_dict, regex=cov_regex, axis='cols')
    return mt1.annotate_cols(__y_w_popstrat = mt1.__y+mt1.__agg_cov)

@typecheck(mt = MatrixTable, 
           str_expr=str)
def clean_fields(mt, str_expr):
    '''Removes fields with names that have str_expr in them'''
    all_fields = list(mt.col)+list(mt.row)+list(mt.entry)+list(mt.globals)
    return mt.drop(*(x for x in all_fields if str_expr in x))

@typecheck(mt=MatrixTable, 
           h2=oneof(nullable(float),
                    nullable(int)),
           starttime=datetime,
           stoptime=datetime,
           runtime=timedelta,
           pi=oneof(float,int),
           is_annot_inf=bool,
           annot_coef_dict=nullable(dict),
           annot_regex=nullable(str),
           h2_normalize=bool,
           is_popstrat=bool,
           cov_coef_dict=nullable(dict),
           cov_regex=nullable(str),
           path_to_save=nullable(str))
def add_sim_description(mt,starttime,stoptime,runtime,h2=None,pi=1,is_annot_inf=False,
                        annot_coef_dict=None, annot_regex=None,h2_normalize=True,
                        is_popstrat=False,cov_coef_dict=None,cov_regex=None,path_to_save=None):
    '''Annotates mt with description of simulation'''
    sim_id = 0
    while (str(sim_id) in [x.strip('sim_desc') for x in list(mt.globals) if 'sim_desc' in x]):
        sim_id += 1
    sim_desc = hl.struct(h2=none_to_null(h2),pi=pi,starttime=str(starttime),
                         stoptime=str(stoptime),runtime=str(runtime),
                         is_annot_inf=is_annot_inf,annot_coef_dict=none_to_null(annot_coef_dict),
                         annot_regex=none_to_null(annot_regex),h2_normalize=h2_normalize, 
                         is_popstrat=is_popstrat,cov_coef_dict=none_to_null(cov_coef_dict),
                         cov_regex=none_to_null(cov_regex),path_to_save=none_to_null(path_to_save))
    mt = mt._annotate_all(global_exprs={f'sim_desc{sim_id}':sim_desc})
    return mt
