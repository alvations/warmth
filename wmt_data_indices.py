#!/usr/bin/env python -*- coding: utf-8 -*-

import io, os, re
from collections import defaultdict
import sys; reload(sys); sys.setdefaultencoding("utf-8")

class WMT14:
    refpath = "data/WMT14/references/newstest2014-ref."
    sysoutpath = "data/WMT14/system-outputs/newstest2014/" 
    scorefile = "data/WMT14/human-2014-05-16.scores"
    langpairs = ['hi-en', 'en-cs', 'en-de', 'en-hi', 'en-ru', 'cs-en', 'en-fr', 'de-en', 'ru-en', 'fr-en']
    systemoutputs = {'fr-en': ['newstest2014.rbmt1.0.fr-en', 'newstest2014.onlineA.0.fr-en', 
                                'newstest2014.uedin-wmt14.3024.fr-en', 'newstest2014.rbmt4.0.fr-en', 
                                'newstest2014.onlineC.0.fr-en', 'newstest2014.Stanford-University.3496.fr-en', 
                                'newstest2014.onlineB.0.fr-en', 'newstest2014.kit.3112.fr-en'], 
                     'en-de': ['newstest2014.uedin-stanford-unconstrained.3539.en-de', 
                                'newstest2014.rbmt4.0.en-de', 'newstest2014.TTT-contrastive.3475.en-de', 
                                'newstest2014.CimS-CORI.3514.en-de', 'newstest2014.Stanford.3469.en-de', 
                                'newstest2014.PROMT-Hybrid.3078.en-de', 'newstest2014.rbmt1.0.en-de', 
                                'newstest2014.UU-Docent.3063.en-de', 'newstest2014.onlineB.0.en-de', 
                                'newstest2014.Stanford.3530.en-de', 'newstest2014.uedin-wmt14.3122.en-de', 
                                'newstest2014.onlineC.0.en-de', 'newstest2014.PROMT-Rule-based.3079.en-de', 
                                'newstest2014.eubridge.3497.en-de', 'newstest2014.uedin-syntax.3285.en-de', 
                                'newstest2014.KIT-primary.3399.en-de', 'newstest2014.onlineA.0.en-de', 
                                'newstest2014.UU.3503.en-de'], 
                     'en-cs': ['newstest2014.CU-TectoMT.2950.en-cs', 'newstest2014.cu-funky.3515.en-cs', 
                                 'newstest2014.cu-depfix.3452.en-cs', 'newstest2014.onlineA.0.en-cs', 
                                 'newstest2014.cu-bojar.3483.en-cs', 'newstest2014.commercial2.3222.en-cs', 
                                 'newstest2014.uedin-wmt14.3021.en-cs', 'newstest2014.uedin-unconstrained.3424.en-cs', 
                                 'newstest2014.onlineB.0.en-cs', 'newstest2014.commercial1.3556.en-cs'], 
                     'cs-en': ['newstest2014.uedin-wmt14.3170.cs-en', 'newstest2014.cu-moses.3383.cs-en', 
                                 'newstest2014.onlineA.0.cs-en', 'newstest2014.uedin-syntax.3289.cs-en', 
                                 'newstest2014.onlineB.0.cs-en'], 
                     'ru-en': ['newstest2014.kaznu1.3549.ru-en', 'newstest2014.rbmt1.0.ru-en', 
                                 'newstest2014.onlineG.0.ru-en', 'newstest2014.uedin-wmt14.3364.ru-en', 
                                 'newstest2014.uedin-syntax.3166.ru-en', 'newstest2014.shad-wmt14.3464.ru-en', 
                                 'newstest2014.rbmt4.0.ru-en', 'newstest2014.onlineA.0.ru-en', 
                                 'newstest2014.PROMT-Rule-based.3085.ru-en', 'newstest2014.onlineB.0.ru-en', 
                                 'newstest2014.PROMT-Hybrid.3084.ru-en', 'newstest2014.AFRL-Post-edited.3431.ru-en', 
                                 'newstest2014.AFRL.3349.ru-en'], 
                     'en-fr': ['newstest2014.rbmt1.0.en-fr', 'newstest2014.uedin-wmt14.3023.en-fr', 
                                 'newstest2014.UU-Docent.3517.en-fr', 'newstest2014.PROMT-Hybrid.3082.en-fr', 
                                 'newstest2014.onlineA.0.en-fr', 'newstest2014.PROMT-Rule-based.3083.en-fr', 
                                 'newstest2014.kit.3440.en-fr', 'newstest2014.dcu-prompsit-ua-rules.3415.en-fr', 
                                 'newstest2014.onlineB.0.en-fr', 'newstest2014.dcu-prompsit-ua.3334.en-fr', 
                                 'newstest2014.onlineC.0.en-fr', 'newstest2014.rbmt4.0.en-fr', 
                                 'newstest2014.UA-Prompsit.3284.en-fr'], 
                     'hi-en': ['newstest2014.uedin-wmt14.3422.hi-en', 'newstest2014.IIIT-Hyderabad.3257.hi-en', 
                                 'newstest2014.iitb-ranked-ppl.3173.hi-en', 'newstest2014.uedin-syntax.3144.hi-en', 
                                 'newstest2014.AFRL.3456.hi-en', 'newstest2014.DCU-HiEn.3564.hi-en', 
                                 'newstest2014.onlineB.0.hi-en', 'newstest2014.CMU.3510.hi-en', 
                                 'newstest2014.onlineA.0.hi-en'], 
                     'en-ru': ['newstest2014.PROMT-Hybrid.3080.en-ru', 'newstest2014.uedin-unconstrained.3445.en-ru', 
                                 'newstest2014.rbmt4.0.en-ru', 'newstest2014.uedin-wmt14.3376.en-ru', 
                                 'newstest2014.onlineB.0.en-ru', 'newstest2014.onlineG.0.en-ru', 
                                 'newstest2014.PROMT-Rule-based.3081.en-ru', 'newstest2014.onlineA.0.en-ru', 
                                 'newstest2014.rbmt1.0.en-ru'], 
                     'en-hi': ['newstest2014.UdS-MaNaWi.3208.en-hi', 'newstest2014.iitb-ranked-ppl.3171.en-hi', 
                                 'newstest2014.DCU-EN-HI.3578.en-hi', 'newstest2014.onlineA.0.en-hi', 
                                 'newstest2014.IPN-UPV-NODEV.3552.en-hi', 'newstest2014.onlineB.0.en-hi', 
                                 'newstest2014.uedin-wmt14.3358.en-hi', 'newstest2014.UDS-MaNaWi-H1-rmOOV.3254.en-hi', 
                                 'newstest2014.IPN-UPV-CONTEXT.3573.en-hi', 'newstest2014.uedin-unconstrained.3360.en-hi', 
                                 'newstest2014.UDS-MaNaWi-H1.3252.en-hi', 'newstest2014.cu-moses.3385.en-hi'], 
                     'de-en': ['newstest2014.kit.3109.de-en', 'newstest2014.onlineA.0.de-en', 
                                 'newstest2014.eubridge.3569.de-en', 'newstest2014.onlineB.0.de-en', 
                                 'newstest2014.uedin-syntax.3035.de-en', 'newstest2014.LIMSI-KIT-Submission.3359.de-en', 
                                 'newstest2014.rbmt4.0.de-en', 'newstest2014.DCU-ICTCAS-Tsinghua-L.3444.de-en', 
                                 'newstest2014.uedin-wmt14.3025.de-en', 'newstest2014.CMU.3461.de-en', 
                                 'newstest2014.onlineC.0.de-en', 'newstest2014.rbmt1.0.de-en', 
                                 'newstest2014.RWTH-primary.3266.de-en']}
    
    def system_outputs(self, langpair):
        return [io.open(self.sysoutpath+langpair+'/'+i,'r').readlines() for i in self.systemoutputs[langpair]]

    def references(self, langpair):
        return io.open(self.refpath+langpair, 'r').readlines()
    
    
WMT = {'14': WMT14()}

for langpair in WMT['14'].langpairs:
    reference = WMT['14'].references(langpair)
    for system_output in WMT['14'].system_outputs(langpair):
        counter = 1
        for ref, sysout in zip(reference, system_output):
            print ref.strip(), sysout.strip()
