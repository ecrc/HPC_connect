pipeline {
/*
 * Defining where to run
 */
//// Any:
// agent any
//gpu-new for a100/v100 GPUs
    agent { label 'gpu-new' }
    triggers {
        pollSCM('H/10 * * * *')
    }

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }

    stages {
        stage ('build') {
            steps {
                sh '''#!/bin/bash -le
                    # Please add your modules here
                         module load cuda/11.4
			 module load mkl
			 module load cmake
		    # build 
                         make
		'''
            }
        }
	   stage ('test') {
            steps {
                sh '''#!/bin/bash -le
                    # Please add your modules here
                         module load cuda/11.4
			 module load mkl
			 module load cmake
		    #Test
                ./GML 100 100 4 100 64 s
		'''
            }
        }
    }
    // Post build actions
    post {
        //always {
        //}
        //success {
        //}
        //unstable {
        //}
        //failure {
        //}
        unstable {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build is UNSTABLE", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
        failure {
                emailext body: "${env.JOB_NAME} - Please go to ${env.BUILD_URL}", subject: "Jenkins Pipeline build FAILED", recipientProviders: [[$class: 'CulpritsRecipientProvider'], [$class: 'RequesterRecipientProvider']]
        }
    }
}
