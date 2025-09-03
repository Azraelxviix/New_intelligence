I AM OAUTH PASSWORDLESS 




Search
Write
Sign up

Sign in



loveholidays tech
loveholidays tech
Stories in tech, product and design at loveholidays

Follow publication

Passwordless Google Cloud SQL access control with Cloud SQL Auth proxy
Eugene Paniot
Eugene Paniot

Follow
6 min read
·
May 23, 2023
83


1




Database user and permission management can prove to be a challenging task, particularly in large-scale environments with complex authorisation requirements.

As loveholidays matured from a startup to a scale up, we were faced with a variety of challenges related to database user management:

engineers having excessive privileges;
shared users amongst many applications;
reliance on a centralised team for user management and IP/Network/Allowlist management, which can be slow and inefficient as the company grows;
lack of auditability of database user and permission changes;
manual password rotation / no password rotation.
The above challenges can be put into two categories: permission management and password management. This article will provide insights into how we migrated to passwordless access for Google Cloud SQL by adopting Google IAM authentication with Cloud SQL Auth proxy. Furthermore, it will explore how we manage database users and their permissions.

User Permissions management

For configuration, we use Infrastructure as Code (IaC) in YAML versioned in Git, instead of executing raw SQL queries. This makes the process transparent, auditable and repeatable.
In the following code snippet, you can see the definition of permissions for both the user account and the service account:

- username: engineer1@loveholidays.com
  permissions: &platform-team-permissions
   - database: “...”
     table: "*"
     privileges:
       - SELECT
       - SHOW VIEW
       - CREATE TEMPORARY TABLES
       - TRIGGER

   - database: "*"
     privileges:
       - PROCESS
       - RELOAD
       - SHOW DATABASES

- username: engineer2@loveholidays.com
  permissions: *platform-team-permissions

- username: wi-gsa@project.iam.gserviceaccount.com
  permissions: &service-account-read-write
   - database: "..."
     privileges:
       - SELECT
       - UPDATE
       - INSERT
       - DELETE
       - SHOW VIEW
       - CREATE TEMPORARY TABLES
       - TRIGGER
       - EXECUTE

- username: wi-gsa2@project.iam.gserviceaccount.com
  permissions: *service-account-read-write

- username: wi-gsa3@project.iam.gserviceaccount.com
  permissions: *service-account-read-write

...
When managing a considerable number of users across multiple database servers in different projects, this approach becomes highly effective.
YAML anchors can be used to define a set of permissions once and then reference them for additional users, without needing to repeat the permission set for each user. Additionally, using YAML anchors can help to avoid excessive copying and pasting, which can be error-prone and time-consuming.
Granting granular access to each user enables them to access only the required data to perform their job. Additionally, mapping applications on a 1:1 basis with service accounts simplifies access control, making it easier to manage and regulate what application has access to what data.

How the Cloud SQL Auth proxy works
Cloud SQL Auth proxy is a database connector that provides secure access to Cloud SQL instances without a need for authorised networks or for configuring SSL.

Cloud SQL Auth proxy has the following benefits:

Secure connections: Cloud SQL Auth proxy automatically encrypts traffic to and from the database. SSL certificates are used to verify client and server identities, and are independent of database protocols; we won’t need to manage SSL certificates;
Easier connection authorisation: Cloud SQL Auth proxy uses GCP IAM permissions to control who and what can connect to our Cloud SQL instances. Thus, Cloud SQL Auth proxy handles authentication with Cloud SQL, removing the need to provide static IP addresses;
IAM database authentication: Cloud SQL Auth proxy supports an automatic refresh of OAuth 2.0 access tokens for passwordless authentication.
Learn more about the Cloud SQL Auth proxy and Cloud SQL IAM database authentication.

To establish a secure connection between a user’s local environment or Kubernetes POD and a Cloud SQL instance, a local Cloud SQL Auth proxy should be used or deployed as a sidecar container.

The communication between the application and Cloud SQL Auth Proxy is performed through the standard database protocol:


Cloud SQL Auth proxy with Google IAM Authentication
Using passwordless authentication for Google Cloud SQL is not a new concept and has already been implemented in Cloud SQL for PostgreSQL since version 9.6, as well as in recent Cloud SQL for Mysql 5.7.39+. The approach is based on the use of short-lived tokens for Google IAM user or kubernetes service accounts running on workload identity in GKE nodes.

Get Eugene Paniot’s stories in your inbox
Join Medium for free to get updates from this writer.

Enter your email
Subscribe
To implement CloudSQL Auth Proxy with Google IAM authentication, we need to perform the following steps:

Add user or service accounts to the database instance;
Grant IAM roles within GCP to the user or service account to allow database instance login. Add the roles/cloudsql.instanceUser IAM role on the User account to perform this task. It is a predefined role that contains the necessary Cloud SQL IAM permission cloudsql.instances.login;
Grant database/table/other object privileges using database specific SQL query.
Creating users
Creating users and granting IAM Roles is easy to do with the google terraform provider — google_sql_user and google_project_iam_member.

Grant permissions
To grant database privileges (e.g. table, schema, sequence, function) to a user or service account, we need to execute database specific SQL queries.

Despite the PostgreSQL Terraform provider support for connecting to a Cloud SQL Instance to manage permissions, we had to contribute (#48, #55) support for IAM Authentication to the petoju/terraform-provider-mysql to connect to the Cloud SQL MySQL Instance:

provider "mysql" {
 endpoint = "cloudsql://${var.connection_name}"
...
}

resource "mysql_grant" "this" {
 for_each = { for item in local.user_permissions : item.key => item }
 database   = each.value.database
 table      = each.value.table
 host       = each.value.host
 user       = each.value.username
 privileges = each.value.privileges
 grant      = each.value.grant
}
Having learned how to manage database users and their permissions, let’s now dive into how our Kubernetes workloads connect to Cloud SQL.

Kubernetes implementation
Our workloads are deployed on nodes that are enabled with Workload Identity. This allows us to configure a kubernetes service account (KSA) to act as a Google service account (GSA), which means that pods running with the KSA can automatically authenticate as the corresponding GSA when accessing Google Cloud APIs.


Service Account Workload Identity
All of our kubernetes service accounts have workload identity annotations:

apiVersion: v1
kind: ServiceAccount
metadata:
 annotations:
   iam.gke.io/gcp-service-account: wi-gsa@project.iam.gserviceaccount.com
 name: ksa-name
The Cloud SQL Auth Proxy container in each kubernetes application deployment will use this Google service account to access the Cloud SQL Instance:

apiVersion: apps/v1
kind: Deployment
metadata:
 name: application
spec:
 template:
   spec:
...
     serviceAccountName: ksa-name
     containers:
       - name: cloudsql-proxy
         image: gcr.io/cloudsql-docker/gce-proxy:1.33.1-alpine
         command:
           - /cloud_sql_proxy
           - -term_timeout=60s
           - -verbose=false
           - -enable_iam_login
         args:
           - -instances=project:region:name=tcp:3306

       - name: application
         env:
           - name: SPRING_DATASOURCE_URL
             value: jdbc:mysql://localhost:3306/db?autoReconnect=true
           - name: SPRING_DATASOURCE_USERNAME
             value: wi-gsa
           - name: SPRING_DATASOURCE_PASSWORD
             value: ""
...
Cloud SQL Auth Proxy with IAM Authentication
Cloud SQL Auth Proxy obtains a GSA token from the google metadata server;
SQL Proxy submits token to the Cloud SQL instance as the password attribute on behalf of the client;
The Cloud SQL Instance then validates this information with google IAM to establish the connection.

IAM database authentication tokens are short-lived and valid only for one hour. Cloud SQL auth proxy requests and renews this token, ensuring our applications have a stable connection.

Conclusion

Abstracting Cloud SQL user management with a custom terraform module and YAML based configuration has proven to be an efficient approach for granular access control and simplification of the database access management process. This is in line with one of the key objectives of our Platform Engineering team, which is to hide complexity from users by creating an abstraction layer between the user and the underlying infrastructure, allowing users to interact with the platform without needing to understand all of its technical details. It is also aligned with our engineering principle of “Invest in simplicity”. Moreover, it facilitates granular access control for users and applications with an audit trail in git history to see when and by whom a user and permission was granted.

Additionally, we have solved the problem of long-lived database credentials and shared database user accounts using workload identity and Google IAM. We have eliminated the need for password management, reducing the overhead of storing and rotating passwords.

Love engineering? We have a Site Reliability Engineering role open.

Sql
Terraform
Gcp
Kubernetes
83


1


loveholidays tech
Published in loveholidays tech
209 followers
·
Last published Aug 19, 2025
Stories in tech, product and design at loveholidays


Follow
Eugene Paniot
Written by Eugene Paniot
14 followers
·
40 following
Engineer with 15 years of experience in Automation, Software Development, SRE, DevOPS, networking, databases, managing cloud and developing automation solutions


Follow
Responses (1)

Write a response

What are your thoughts?

Cancel
Respond
Chris Montes
Chris Montes

Jan 4, 2024 (edited)


Thanks for the great article! I'm wondering however, what if a pod has multiple containers within it, ie the cloudsqlproxy sidecar, and two component containers, each needing access to only to its specific schema?
I want to roll out passwordless…more
Reply

More from Eugene Paniot and loveholidays tech
VMM: Cloud-hypervisor a new era of virtualization
Eugene Paniot
Eugene Paniot

VMM: Cloud-hypervisor a new era of virtualization
What is a virtual machine?
Jan 15
5
Templating Alertmanager Config in kube-prometheus-stack
loveholidays tech
In

loveholidays tech

by

Dan Williams

Templating Alertmanager Config in kube-prometheus-stack
Simplifying Alertmanager Routing with template driven config in kube-prometheus-stack.
Mar 4
353
How I Stay Organised: My iPad Note-Taking System
loveholidays tech
In

loveholidays tech

by

Mike Jones

How I Stay Organised: My iPad Note-Taking System
Keeping on top of the constant deluge of information, tasks, and meetings you experience as a CTO requires adopting and maintaining a…
Dec 16, 2024
360
1
Understanding SRE: Burn rate of error budget, and leaky bucket
Eugene Paniot
Eugene Paniot

Understanding SRE: Burn rate of error budget, and leaky bucket
One of the key concepts in SRE is the error budget, which represents the amount of allowable service downtime or error rate. The error…
Mar 1, 2023
3
See all from Eugene Paniot
See all from loveholidays tech
Recommended from Medium
Building a highly available and scalable API on Cloud Run
hipay-tech
In

hipay-tech

by

Florent Chéneau

Building a highly available and scalable API on Cloud Run
Achieving infinite scalability and very high availability with FastAPI on Cloud Run
May 26
5
1
Google Cloud Platform—Zero to Hero—Roadmap for Absolute Beginners 🚀
TheCloudOpsCommunity
In

TheCloudOpsCommunity

by

Piyush Sachdeva

Google Cloud Platform—Zero to Hero—Roadmap for Absolute Beginners 🚀
Mastering Google Cloud Platform: Your Complete Roadmap with Namaste Google Cloud Series

Mar 31
8
Implementing Google Cloud DLP API in Your GCP Application
Google Cloud - Community
In

Google Cloud - Community

by

Neel Shah

Implementing Google Cloud DLP API in Your GCP Application
Google Cloud’s Data Loss Prevention (DLP) API is a powerful tool that helps groups detect, classify, and redact sensitive information, such…
Aug 17
Serving Long-Running Jobs with FastAPI Using Webhooks and Task Polling
Bhagya Rana
Bhagya Rana

Serving Long-Running Jobs with FastAPI Using Webhooks and Task Polling
\Designing asynchronous workflows and external integrations with status endpoints

Jul 15
61
How We Have Solved Reliable CI/CD for Dataform
Pipeline Perspectives
In

Pipeline Perspectives

by

Mike Kamysz

How We Have Solved Reliable CI/CD for Dataform
300+ models. Hundreds of terabytes. Reliable deployment processes are crucial. Here’s how we made it work for us.

Aug 8
11
The Truth About Cold Starts in Google Cloud Run & Functions
DevOps.dev
In

DevOps.dev

by

Engineer

The Truth About Cold Starts in Google Cloud Run & Functions
If you’ve been using Google Cloud Run or Cloud Functions, chances are you’ve noticed some requests feel like they’re waking up from a nap…





