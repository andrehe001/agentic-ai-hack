
# 1. Environment Creation and Resources Deployment

Welcome to your very first challenge! Your goal in this challenge is to create the services and enviornment necessary to conduct this hackathon. You will deploy the required resources in Azure, create your development enviornment and all the assets necessary for the subsequent challenges. By completing this challenge, you will set up the foundation for the rest of the hackathon. 

If something is not working correctly, please do let your coach know!


## 1.1 Preparation

Before you start, please fork this repository to your GitHub account by clicking the `Fork` button in the upper right corner of the repository's main screen (or follow the [documentation](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo#forking-a-repository)). This will allow you to make changes to the repository and save your progress.

## 1.2 Resource Deployment Guide
The first step on this hackathon will be to create the resources we will use throughout the day. Clicking on button bellow will redirect you to the Azure portal to deploy the resources using the [ARM template](iac) provided in this repository.

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fmartaldsantos%2Fdoc-process-hack%2Fmain%2FChallenge1%2Fiac%2Fazuredeploy.json)

**NOTE:** Deployment may fail if the resource provider `Microsoft.AlertsManagement` is not registered in your. Follow the [documentation](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/resource-providers-and-types#register-resource-provider-1) to register it and the re-run the deployment.

Resource deployment can take up to 5 minutes. Once the deployment is complete, you will see the resources in your Azure portal.

In the meantime, you can proceed with the next step - opening pre-configured development environment in GitHub Codespaces.

## 1.3 Development Environment

GitHub Codespaces is a cloud-based development environment that allows you to code from anywhere. It provides a fully configured environment that can be launched directly from any GitHub repository, saving you from lengthy setup times. You can access Codespaces from your browser, Visual Studio Code, or the GitHub CLI, making it easy to work from virtually any device.

To open GitHub Codespaces, click on the button below:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/)

Please select your forked repository from the dropdown and, if necessary, adjust other settings of GitHub Codespace.

**NOTE:** If GitHub Codespaces is not enabled in your organization, you can enable it by following the instructions [here](https://docs.github.com/en/codespaces/managing-codespaces-for-your-organization/enabling-or-disabling-github-codespaces-for-your-organization), or, if you cannot change your GitHub organization's settings, create free personal GitHub account [here](https://github.com/signup?ref_cta=Sign+up&ref_loc=header+logged+out&ref_page=%2F&source=header-home).

## 1.4 Verify the creation of your resources

Go back to your `Azure Portal` and find your `Resource Group`that should by now contain 9 resources and look like this:

![image](https://github.com/user-attachments/assets/e04298dd-a601-47a2-8fda-bd0cac19f313)



## 1.5 Let's retrieve the necessary keys


After deploying the resources, you will need to configure the environment variables in the `.env` file. The `.env` file is a configuration file that contains the environment variables for the application. The `.env` file is automatically created running the following command within the terminal in your Codespace:

```bash
cd Challenge1
./get-keys.sh --resource-group <resource-group-name>
```

This script will connect to Azure and fetch the necessary keys and populate the `.env` file with the required values in the root directory of the repository. If needed, script will prompt you to sign in to your Azure account.

## 1.6 Verify `.env` setup

When the script is finished, review the `.env` file to ensure that all the values are correct. If you need to make any changes, you can do so manually.

The default sample has an `.env.sample` file that shows the relevant environment variables that need to be configured in this project. The script should create a `.env` file that has these same variables _but populated with the right values_ for your Azure resources.

If the file is not created, simply copy over `.env.sample` to `.env` - then populate those values manually from the respective Azure resource pages using the Azure Portal.

## Conclusion
By reaching this section you should have every resource and installed the requirements necessary to conduct the hackathon. You have deployed an Azure AI Document Intelligence service, an Azure Cosmos DB account, an Azure Storage Account and Azure OpenAI Service.

In the next challenges, you will use these services to build a strong document processing workflow.

Now the real fun begins!