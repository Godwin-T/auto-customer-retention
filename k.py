# from app.churn_guard.guard_deploy.app import app


# if __name__ == "__main__":
#     app.run(debug=True, port=9696)

from app.churn_guard.prefect.deployment_flow import main

if __name__ == "__main__":

    main.serve(name="data=processing-deployment", cron="*/3 * * * *")
# if __name__ == "__main__":
#     main.deploy(
#         name="my-deployment",
#         work_pool_name="customer-retention",
#         image="freshinit/fresh:V1",
#         push = False,
#         cron="*/3 * * * *"
#     )
