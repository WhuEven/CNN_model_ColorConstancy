# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:00:05 2018

@author: phamh
"""
#import libraries
import progressbar as pb

#define progress timer class
class progress_timer:

    def __init__(self, n_iter, description="Something"):
        self.n_iter         = n_iter
        self.iter           = 0
        self.description    = description + ': '
        self.timer          = None
        self.initialize()

    def initialize(self):
        #initialize timer
        widgets = [self.description, pb.Percentage(), ' ',   
                   pb.Bar('=', '[', ']'), ' ', pb.ETA()]
        self.timer = pb.ProgressBar(widgets=widgets, maxval=self.n_iter).start()

    def update(self, q=1):
        #update timer
        self.timer.update(self.iter)
        self.iter += q

    def finish(self):
        #end timer
        self.timer.finish()
        
        
# =============================================================================
# #initialize
# pt = progress_timer(description= 'For loop example', n_iter=1000000)
# #for loop example
# for i in range(0,1000000):
#     #update
#     pt.update()
# #finish
# pt.finish()
# =============================================================================

