# Copyright (C) 2016 Henrique Pereira Coutada Miranda, Alejandro Molina-Sanchez
# All rights reserved.
#
# This file is part of yambopy
#
#
import subprocess
from schedulerpy import *
from textwrap import dedent
from copy import deepcopy
from collections import OrderedDict

class Pbs(Scheduler):
    """
    Class to submit jobs through the PBS scheduler
    _vardict states the default assignement of the nodes and cores variables
    from the schduler class to the variables needed in this class
    """
    _vardict = {"cores":"core",
                "nodes":"select"}
                   
    def initialize(self):
        self.get_vardict()
        args = self.arguments
        queue = self.get_arg("queue")
        if self.name: args.append("-N %s"%self.name)
        
        if queue: args.append("-q %s"%(queue))
        group_list = self.get_arg("group_list")
        
        if group_list: args.append("-W group_list=%s"%group_list)
        dependent = self.get_arg("dependent")
        
        if dependent: args.append("-W depend=afterok:%s"%dependent)
        args.append("-l walltime=%s"%self.walltime)
            
        resources_line = self.get_resources_line()
        if resources_line:
            args.append("-l %s"%resources_line)
        
    def get_resources_line(self):
        """
        get the the line with the resources
        """
        tags = ['select','nodes','core','ppn','ncpus','mpiprocs','ompthreads']
        args = [self.get_arg(tag) for tag in tags]
        resources = OrderedDict([(tag,value) for tag,value in zip(tags,args) if value is not None])
        if self.nodes: resources[self.vardict['nodes']] = self.nodes
        if self.cores: resources[self.vardict['cores']] = self.cores
        
        # memory stuff
        mem = self.get_arg("mem")
        if mem:
            mem = int(mem)
            if self.cores: mem *= self.cores
            if self.nodes: mem *= self.nodes
            resources["vmem"] = "%dMB"%mem
        
        resources_line = ":".join(["%s=%s"%(item,value) for item,value in resources.items()])
        
        return resources_line
    
    def get_script(self):
        """
        get a .pbs file to be submitted using qsub
        qsub <filename>.pbs
        """
        s = '#!/bin/bash\n'
        s += "\n".join(["#PBS %s"%s for s in self.arguments])+'\n'
        s += self.get_commands()
        return s
        
    def get_bash(self):
        """
        get a bash command to submit the job
        """
        s = "echo \"%s\" | "%self.get_commands().replace("'","\"").replace("\"","\\\"")
        s += "qsub \\\n"
        s += " \\\n".join(self.arguments)
        return s
        
    def __str__(self):
        return self.get_script()
        
    def run(self,dry=False,silent=True):
        """
        run the command
        arguments:
        dry - only print the commands to be run on the screen
        """
        command = self.get_bash()
        
        if dry:
            print command
        else:
            p = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True,executable='/bin/bash')
            self.stdout,self.stderr = p.communicate()
            
            #check if there is stderr
            if self.stderr: raise Exception(self.stderr)
            
            #check if there is stdout
            if not silent: print self.stdout
            
            #get jobid
            self.jobid = self.stdout.split('\n')[0]
            print "jobid:",self.jobid

        