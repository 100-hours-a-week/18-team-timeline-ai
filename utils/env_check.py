def check_env_vars(vars_dict, logger, prefix=""):
    for key, value in vars_dict.items():
        if not value:
            logger.error(f"{prefix}{key} 환경변수가 설정되지 않았습니다.")
