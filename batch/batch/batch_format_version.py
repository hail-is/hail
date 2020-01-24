from hailtop.batch_client.aioclient import Job


class BatchFormatVersion:
    def __init__(self, format_version):
        self.format_version = format_version

    def has_full_spec_in_gcs(self):
        return self.format_version > 1

    def has_full_status_in_gcs(self):
        return self.format_version > 1

    def has_full_status_in_db(self):
        return self.format_version == 1

    def has_attempt_in_log_path(self):
        return self.format_version > 1

    def db_spec(self, spec):
        if self.format_version == 1:
            return spec

        return {
            'secrets': spec.get('secrets'),
            'service_account': spec.get('service_account'),
            'has_input_files': len(spec.get('input_files', [])) > 0,
            'has_output_files': len(spec.get('output_files', [])) > 0,
            'attributes': spec.get('attributes')
        }

    def get_spec_has_input_files(self, spec):
        if self.format_version == 1:
            return len(spec.get('input_files', [])) > 0
        return spec['has_input_files']

    def get_spec_has_output_files(self, spec):
        if self.format_version == 1:
            return len(spec.get('output_files', [])) > 0
        return spec['has_output_files']

    def db_status(self, status):
        if self.format_version == 1:
            return status

        job_status = {'status': status}
        return {
            'exit_code': Job.exit_code(job_status),
            'duration': Job.total_duration_msecs(job_status)
        }
