#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:17:46 2024

@author: sascha
"""

import yaml

def remove_build_strings(yaml_file, output_file):
    # Load the YAML file
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)

    # Check and modify the dependencies if they exist
    if 'dependencies' in data:
        new_dependencies = []
        for dep in data['dependencies']:
            # Split the dependency string and remove the build string if present
            if '=' in dep:
                package_version = dep.split('=')[0:2]  # Keep only name and version
                new_dependencies.append('='.join(package_version))
            else:
                new_dependencies.append(dep)  # Keep as is if no '='
        data['dependencies'] = new_dependencies

    # Write the modified data to the output file
    with open(output_file, 'w') as file:
        yaml.safe_dump(data, file)

# Example usage
remove_build_strings('hsssm_env.yaml', 'hsssm_env_mod.yaml')