create schema if not exists metadata;
create schema if not exists bronze;
create schema if not exists silver;
create schema if not exists gold;

create table if not exists metadata.source (
    source_id text primary key,
    source_name text not null,
    operator_name text,
    source_url text,
    access_class text not null,
    license_name text,
    contains_person_level_data boolean not null default false,
    credential_status text not null,
    first_registered_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb
);

create table if not exists metadata.harvest_run (
    harvest_run_id bigint generated always as identity primary key,
    started_at timestamptz not null,
    completed_at timestamptz,
    pipeline_version text not null,
    status text not null,
    configuration jsonb not null,
    host_metadata jsonb not null default '{}'::jsonb
);

create table if not exists metadata.asset (
    asset_id bigint generated always as identity primary key,
    source_id text not null references metadata.source(source_id),
    harvest_run_id bigint references metadata.harvest_run(harvest_run_id),
    dataset_name text not null,
    request_url text,
    request_method text,
    request_parameters jsonb not null default '{}'::jsonb,
    retrieved_at timestamptz,
    http_status integer,
    content_type text,
    byte_count bigint not null,
    sha256 text not null,
    absolute_path text not null,
    status text not null,
    error_message text,
    created_at timestamptz not null default now(),
    unique (source_id, request_url, absolute_path, sha256)
);

create table if not exists metadata.dataset (
    dataset_id bigint generated always as identity primary key,
    source_id text not null references metadata.source(source_id),
    asset_id bigint references metadata.asset(asset_id),
    dataset_name text not null,
    medallion_layer text not null,
    storage_format text not null,
    absolute_path text not null,
    row_count bigint not null default 0,
    column_count integer not null default 0,
    schema_json jsonb not null default '{}'::jsonb,
    sha256 text not null,
    loaded_at timestamptz not null default now(),
    unique (medallion_layer, absolute_path, sha256)
);

alter table metadata.dataset add column if not exists data_role text not null default 'unclassified';
alter table metadata.dataset add column if not exists modeling_eligible boolean not null default false;

update metadata.dataset
set
    data_role = case
        when dataset_name in (
            'canonical_cases',
            'malaria_puskesmas_indicators',
            'open_dashboard_records',
            'puskesmas_cases',
            'public_enriched_observations',
            'skdr_weekly_indicators',
            'vector_surveillance_indicators',
            'who_malaria_indicators'
        ) then 'model_ready_measurement_rows'
        when dataset_name in ('public_tabular_cells', 'surveillance_portal_cells')
            then 'source_table_cells'
        when dataset_name ~ '(diagnostic|predictability|frontier|hotspot|cluster|decomposition|similarity|multiscale|seasonal|spatial_boundary)'
            then 'analytical_output'
        when dataset_name ~ '(status|catalog|manifest|resource|literature|document_page|normalization|quality|readiness|reconciliation|audit|candidate)'
            then 'catalog_provenance_or_review'
        when dataset_name ~ '(occurrence|sequence|boundary|document_text)'
            then 'scientific_context'
        else 'other_supporting_data'
    end,
    modeling_eligible = dataset_name in (
        'canonical_cases',
        'malaria_puskesmas_indicators',
        'open_dashboard_records',
        'puskesmas_cases',
        'public_enriched_observations',
        'skdr_weekly_indicators',
        'vector_surveillance_indicators',
        'who_malaria_indicators'
    );

create table if not exists metadata.blocked_source (
    source_id text primary key,
    access_class text not null,
    reason text not null,
    endpoint text not null,
    verified_at timestamptz not null default now()
);

create table if not exists bronze.record (
    bronze_record_id bigint generated always as identity primary key,
    dataset_id bigint not null references metadata.dataset(dataset_id),
    row_number bigint not null,
    source_row_hash text not null,
    payload jsonb not null,
    ingested_at timestamptz not null default now(),
    unique (dataset_id, row_number, source_row_hash)
);

create index if not exists bronze_record_dataset_idx on bronze.record(dataset_id);
create index if not exists bronze_record_hash_idx on bronze.record(source_row_hash);
create index if not exists bronze_record_payload_idx on bronze.record using gin(payload jsonb_path_ops);

create table if not exists silver.observation (
    silver_observation_id bigint generated always as identity primary key,
    bronze_record_id bigint not null references bronze.record(bronze_record_id),
    dataset_id bigint not null references metadata.dataset(dataset_id),
    source_id text not null references metadata.source(source_id),
    disease text,
    metric text not null,
    value_numeric double precision,
    value_text text,
    unit text,
    period_start date,
    period_end date,
    temporal_resolution text,
    admin_0 text,
    admin_1 text,
    admin_2 text,
    admin_3 text,
    admin_4 text,
    spatial_resolution text,
    facility_name text,
    facility_type text,
    sex text,
    age_group text,
    species text,
    case_definition text,
    confirmation_status text,
    privacy_class text not null default 'aggregate',
    provenance jsonb not null,
    record_hash text not null,
    processed_at timestamptz not null default now(),
    unique (record_hash)
);

alter table silver.observation add column if not exists admin_4 text;
alter table silver.observation add column if not exists spatial_resolution text;

create index if not exists silver_observation_period_idx on silver.observation(period_start);
create index if not exists silver_observation_geo_idx on silver.observation(admin_1, admin_2, admin_3);
create index if not exists silver_observation_admin4_idx on silver.observation(admin_4);
create index if not exists silver_observation_metric_idx on silver.observation(disease, metric);
create index if not exists silver_observation_facility_idx on silver.observation(facility_name);

create table if not exists metadata.lineage (
    lineage_id bigint generated always as identity primary key,
    bronze_record_id bigint references bronze.record(bronze_record_id),
    silver_observation_id bigint references silver.observation(silver_observation_id),
    gold_fact_id bigint,
    transformation_name text not null,
    transformation_version text not null,
    executed_at timestamptz not null default now(),
    parameters jsonb not null default '{}'::jsonb
);

create table if not exists metadata.quality_result (
    quality_result_id bigint generated always as identity primary key,
    dataset_id bigint references metadata.dataset(dataset_id),
    layer_name text not null,
    check_name text not null,
    severity text not null,
    passed boolean not null,
    observed_value double precision,
    expected_value text,
    details jsonb not null default '{}'::jsonb,
    checked_at timestamptz not null default now()
);

create table if not exists gold.surveillance_fact (
    gold_fact_id bigint generated always as identity primary key,
    silver_observation_id bigint not null references silver.observation(silver_observation_id),
    disease text,
    metric text not null,
    value double precision,
    unit text,
    period_start date,
    period_end date,
    temporal_resolution text,
    admin_0 text,
    admin_1 text,
    admin_2 text,
    admin_3 text,
    admin_4 text,
    spatial_resolution text,
    facility_name text,
    facility_type text,
    sex text,
    age_group text,
    species text,
    case_definition text,
    confirmation_status text,
    source_id text not null,
    source_dataset_id bigint not null,
    provenance jsonb not null,
    fact_hash text not null unique,
    published_at timestamptz not null default now()
);

alter table gold.surveillance_fact add column if not exists admin_4 text;
alter table gold.surveillance_fact add column if not exists spatial_resolution text;

alter table metadata.lineage drop constraint if exists lineage_gold_fact_fk;
alter table metadata.lineage add constraint lineage_gold_fact_fk foreign key (gold_fact_id) references gold.surveillance_fact(gold_fact_id);

create index if not exists gold_fact_time_idx on gold.surveillance_fact(period_start);
create index if not exists gold_fact_geo_idx on gold.surveillance_fact(admin_1, admin_2, admin_3);
create index if not exists gold_fact_admin4_idx on gold.surveillance_fact(admin_4);
create index if not exists gold_fact_model_idx on gold.surveillance_fact(disease, metric, temporal_resolution);

create or replace view gold.native_observation as
select
    g.gold_fact_id as observation_id,
    g.silver_observation_id,
    g.source_id,
    g.source_dataset_id,
    d.dataset_name as source_dataset_name,
    g.disease,
    g.metric,
    g.value,
    g.unit,
    g.period_start,
    g.period_end,
    g.temporal_resolution,
    case
        when g.period_start is not null then
            coalesce(
                g.period_end,
                case lower(trim(g.temporal_resolution))
                    when 'week' then g.period_start + 6
                    when 'month' then
                        (date_trunc('month', g.period_start) + interval '1 month - 1 day')::date
                    when 'quarter' then
                        (date_trunc('quarter', g.period_start) + interval '3 months - 1 day')::date
                    when 'year' then
                        (date_trunc('year', g.period_start) + interval '1 year - 1 day')::date
                    else g.period_start
                end
            ) - g.period_start + 1
    end as temporal_span_days,
    case
        when nullif(trim(g.spatial_resolution), '') is not null then g.spatial_resolution
        when lower(coalesce(
            nullif(g.provenance #>> '{source_record,aggregation_level}', ''),
            nullif(g.provenance #>> '{source_record,admin_level}', '')
        )) in
            ('kelurahan', 'desa', 'village') then 'adm4'
        when lower(coalesce(
            nullif(g.provenance #>> '{source_record,aggregation_level}', ''),
            nullif(g.provenance #>> '{source_record,admin_level}', '')
        )) in
            ('kecamatan', 'subdistrict') then 'adm3'
        when lower(coalesce(
            nullif(g.provenance #>> '{source_record,aggregation_level}', ''),
            nullif(g.provenance #>> '{source_record,admin_level}', '')
        )) in
            ('puskesmas', 'hospital', 'facility') then 'facility'
        when lower(coalesce(
            nullif(g.provenance #>> '{source_record,aggregation_level}', ''),
            nullif(g.provenance #>> '{source_record,admin_level}', '')
        )) in ('admin0', 'adm0') then 'adm0'
        when lower(coalesce(
            nullif(g.provenance #>> '{source_record,aggregation_level}', ''),
            nullif(g.provenance #>> '{source_record,admin_level}', '')
        )) in ('admin1', 'adm1') then 'adm1'
        when lower(coalesce(
            nullif(g.provenance #>> '{source_record,aggregation_level}', ''),
            nullif(g.provenance #>> '{source_record,admin_level}', '')
        )) in ('admin2', 'adm2') then 'adm2'
        when lower(coalesce(
            nullif(g.provenance #>> '{source_record,aggregation_level}', ''),
            nullif(g.provenance #>> '{source_record,admin_level}', '')
        )) in ('admin3', 'adm3') then 'adm3'
        when nullif(trim(g.facility_name), '') is not null then 'facility'
        when nullif(trim(g.admin_4), '') is not null then 'adm4'
        when nullif(trim(g.admin_3), '') is not null then 'adm3'
        when nullif(trim(g.admin_2), '') is not null then 'adm2'
        when nullif(trim(g.admin_1), '') is not null then 'adm1'
        when nullif(trim(g.admin_0), '') is not null then 'adm0'
        else 'unspecified'
    end as spatial_resolution,
    case
        when nullif(trim(g.spatial_resolution), '') is not null then g.spatial_resolution
        when coalesce(
            nullif(lower(g.provenance #>> '{source_record,aggregation_level}'), ''),
            nullif(lower(g.provenance #>> '{source_record,admin_level}'), '')
        ) is not null then coalesce(
            nullif(lower(g.provenance #>> '{source_record,aggregation_level}'), ''),
            nullif(lower(g.provenance #>> '{source_record,admin_level}'), '')
        )
        when nullif(trim(g.facility_name), '') is not null
            then coalesce(nullif(lower(g.facility_type), ''), 'facility')
        when nullif(trim(g.admin_4), '') is not null then 'adm4'
        when nullif(trim(g.admin_3), '') is not null then 'adm3'
        when nullif(trim(g.admin_2), '') is not null then 'adm2'
        when nullif(trim(g.admin_1), '') is not null then 'adm1'
        when nullif(trim(g.admin_0), '') is not null then 'adm0'
        else 'unspecified'
    end as spatial_unit_type,
    g.admin_0,
    g.admin_1,
    g.admin_2,
    g.admin_3,
    g.facility_name,
    g.facility_type,
    g.sex,
    g.age_group,
    g.species,
    g.case_definition,
    g.confirmation_status,
    g.provenance,
    g.fact_hash,
    g.published_at,
    lower(nullif(trim(g.temporal_resolution), '')) as temporal_resolution_normalized,
    g.period_end as period_end_reported,
    coalesce(
        g.period_end,
        case lower(trim(g.temporal_resolution))
            when 'week' then g.period_start + 6
            when 'month' then
                (date_trunc('month', g.period_start) + interval '1 month - 1 day')::date
            when 'quarter' then
                (date_trunc('quarter', g.period_start) + interval '3 months - 1 day')::date
            when 'year' then
                (date_trunc('year', g.period_start) + interval '1 year - 1 day')::date
            else g.period_start
        end
    ) as period_end_effective,
    nullif(g.provenance #>> '{source_record,subunit}', '') as source_subunit,
    coalesce(
        nullif(lower(g.provenance #>> '{source_record,aggregation_level}'), ''),
        nullif(lower(g.provenance #>> '{source_record,admin_level}'), '')
    ) as source_aggregation_level,
    nullif(g.provenance #>> '{source_record,source}', '') as upstream_source,
    nullif(g.provenance #>> '{source_record,source_record_id}', '')
        as upstream_source_record_id,
    g.admin_4
from gold.surveillance_fact g
join metadata.dataset d on d.dataset_id = g.source_dataset_id;

drop view if exists gold.modeling_series;

create view gold.modeling_series as
select *
from gold.native_observation
where value is not null and period_start is not null;

drop view if exists gold.aggregated_series;

create view gold.aggregated_series as
select
    source_id,
    source_dataset_id,
    source_dataset_name,
    upstream_source,
    disease,
    metric,
    unit,
    temporal_resolution,
    period_start,
    period_end,
    spatial_resolution,
    spatial_unit_type,
    admin_0,
    admin_1,
    admin_2,
    admin_3,
    admin_4,
    facility_name,
    facility_type,
    sex,
    age_group,
    species,
    case_definition,
    confirmation_status,
    sum(value) as value,
    count(*) as contributing_records,
    array_agg(observation_id order by observation_id) as observation_ids
from gold.native_observation
where value is not null and period_start is not null
group by source_id, source_dataset_id, source_dataset_name, upstream_source, disease, metric, unit,
    temporal_resolution, period_start, period_end, spatial_resolution, spatial_unit_type,
    admin_0, admin_1, admin_2, admin_3, admin_4, facility_name, facility_type, sex, age_group,
    species, case_definition, confirmation_status;

drop view if exists gold.grain_inventory;

create view gold.grain_inventory as
select
    source_id,
    source_dataset_id,
    source_dataset_name,
    upstream_source,
    disease,
    metric,
    spatial_resolution,
    spatial_unit_type,
    temporal_resolution_normalized as temporal_resolution,
    count(*) as observations,
    min(period_start) as first_period,
    max(period_end_effective) as last_period,
    count(distinct admin_1) filter (where admin_1 is not null) as provinces,
    count(distinct admin_2) filter (where admin_2 is not null) as districts,
    count(distinct admin_3) filter (where admin_3 is not null) as subdistricts,
    count(distinct admin_4) filter (where admin_4 is not null) as villages,
    count(distinct facility_name) filter (where facility_name is not null) as facilities,
    count(distinct source_subunit) filter (where source_subunit is not null) as source_subunits
from gold.native_observation
group by source_id, source_dataset_id, source_dataset_name, upstream_source, disease, metric,
    spatial_resolution, spatial_unit_type, temporal_resolution_normalized;

drop view if exists gold.source_coverage;

create view gold.source_coverage as
select
    source_id,
    disease,
    metric,
    temporal_resolution,
    case
        when facility_name is not null then 'facility'
        when admin_4 is not null then 'adm4'
        when admin_3 is not null then 'adm3'
        when admin_2 is not null then 'adm2'
        when admin_1 is not null then 'adm1'
        when admin_0 is not null then 'adm0'
        else 'unspecified'
    end as spatial_resolution,
    min(period_start) as first_period,
    max(period_end) as last_period,
    count(*) as observations,
    count(distinct facility_name) filter (where facility_name is not null) as facilities,
    count(distinct admin_1) filter (where admin_1 is not null) as provinces,
    count(distinct admin_2) filter (where admin_2 is not null) as districts
from gold.surveillance_fact
group by 1, 2, 3, 4, 5;

create or replace view metadata.asset_provenance as
select
    a.asset_id,
    a.source_id,
    s.source_name,
    a.dataset_name,
    a.request_url,
    a.retrieved_at,
    a.http_status,
    a.content_type,
    a.byte_count,
    a.sha256,
    a.absolute_path,
    a.status,
    a.error_message
from metadata.asset a
join metadata.source s using (source_id);

create table if not exists metadata.diagnostic_run (
    diagnostic_run_id uuid primary key,
    started_at timestamptz not null,
    completed_at timestamptz,
    status text not null,
    input_relation text not null,
    input_snapshot_sha256 text not null,
    code_version text not null,
    configuration jsonb not null,
    software_environment jsonb not null,
    run_metadata jsonb not null default '{}'::jsonb
);

create table if not exists metadata.diagnostic_method (
    method_id text primary key,
    method_family text not null,
    method_name text not null,
    method_version text not null,
    minimum_periods integer,
    requires_regular_time boolean not null default false,
    requires_multiple_locations boolean not null default false,
    requires_geometry boolean not null default false,
    description text not null,
    citation_url text,
    default_parameters jsonb not null default '{}'::jsonb
);

create table if not exists metadata.diagnostic_series (
    diagnostic_run_id uuid not null references metadata.diagnostic_run(diagnostic_run_id),
    series_id text not null,
    panel_id text not null,
    source_id text not null,
    source_dataset_id bigint not null,
    source_dataset_name text not null,
    upstream_source text,
    disease text,
    metric text not null,
    unit text,
    spatial_resolution text not null,
    spatial_unit_type text not null,
    temporal_resolution text not null,
    admin_0 text,
    admin_1 text,
    admin_2 text,
    admin_3 text,
    facility_name text,
    facility_type text,
    source_subunit text,
    source_aggregation_level text,
    first_period date,
    last_period date,
    observations integer not null,
    expected_periods integer,
    completeness double precision,
    duplicate_periods integer not null,
    regular_time boolean not null,
    eligibility jsonb not null,
    primary key (diagnostic_run_id, series_id)
);

create table if not exists metadata.diagnostic_input (
    diagnostic_run_id uuid not null references metadata.diagnostic_run(diagnostic_run_id),
    series_id text not null,
    observation_id bigint not null references gold.surveillance_fact(gold_fact_id),
    primary key (diagnostic_run_id, series_id, observation_id),
    foreign key (diagnostic_run_id, series_id)
        references metadata.diagnostic_series(diagnostic_run_id, series_id)
);

create table if not exists gold.diagnostic_statistic (
    diagnostic_statistic_id bigint generated always as identity primary key,
    diagnostic_run_id uuid not null references metadata.diagnostic_run(diagnostic_run_id),
    series_id text,
    panel_id text,
    method_id text not null references metadata.diagnostic_method(method_id),
    statistic_name text not null,
    estimate double precision,
    p_value double precision,
    q_value double precision,
    confidence_lower double precision,
    confidence_upper double precision,
    unit text,
    status text not null,
    interpretation text,
    parameters jsonb not null default '{}'::jsonb,
    result_hash text not null unique,
    created_at timestamptz not null default now()
);

create table if not exists gold.diagnostic_component (
    diagnostic_component_id bigint generated always as identity primary key,
    diagnostic_run_id uuid not null references metadata.diagnostic_run(diagnostic_run_id),
    series_id text,
    panel_id text,
    method_id text not null references metadata.diagnostic_method(method_id),
    component_name text not null,
    component_index integer,
    coordinate_name text,
    coordinate_value double precision,
    period_start date,
    location_name text,
    value double precision,
    component_metadata jsonb not null default '{}'::jsonb
);

create table if not exists gold.diagnostic_event (
    diagnostic_event_id bigint generated always as identity primary key,
    diagnostic_run_id uuid not null references metadata.diagnostic_run(diagnostic_run_id),
    series_id text,
    panel_id text,
    method_id text not null references metadata.diagnostic_method(method_id),
    period_start date,
    period_end date,
    location_name text,
    event_type text not null,
    score double precision,
    threshold double precision,
    direction text,
    severity text,
    p_value double precision,
    q_value double precision,
    event_metadata jsonb not null default '{}'::jsonb
);

create table if not exists metadata.diagnostic_artifact (
    artifact_id uuid primary key,
    diagnostic_run_id uuid not null references metadata.diagnostic_run(diagnostic_run_id),
    series_id text,
    panel_id text,
    method_id text references metadata.diagnostic_method(method_id),
    artifact_role text not null,
    media_type text not null,
    absolute_path text not null,
    relative_path text not null,
    sha256 text not null,
    byte_count bigint not null,
    width_pixels integer,
    height_pixels integer,
    created_at timestamptz not null,
    provenance jsonb not null,
    unique (diagnostic_run_id, relative_path, sha256)
);

create index if not exists diagnostic_series_panel_idx
    on metadata.diagnostic_series(diagnostic_run_id, panel_id);
create index if not exists diagnostic_statistic_series_idx
    on gold.diagnostic_statistic(diagnostic_run_id, series_id, method_id);
create index if not exists diagnostic_statistic_panel_idx
    on gold.diagnostic_statistic(diagnostic_run_id, panel_id, method_id);
create index if not exists diagnostic_component_series_idx
    on gold.diagnostic_component(diagnostic_run_id, series_id, method_id);
create index if not exists diagnostic_event_series_idx
    on gold.diagnostic_event(diagnostic_run_id, series_id, method_id, period_start);
create index if not exists diagnostic_artifact_run_idx
    on metadata.diagnostic_artifact(diagnostic_run_id, artifact_role);

create or replace view gold.latest_diagnostic_statistic as
select s.*
from gold.diagnostic_statistic s
join (
    select diagnostic_run_id
    from metadata.diagnostic_run
    where status = 'completed'
    order by completed_at desc
    limit 1
) r using (diagnostic_run_id);

create or replace view gold.latest_diagnostic_event as
select e.*
from gold.diagnostic_event e
join (
    select diagnostic_run_id
    from metadata.diagnostic_run
    where status = 'completed'
    order by completed_at desc
    limit 1
) r using (diagnostic_run_id);

create table if not exists metadata.predictability_run (
    predictability_run_id uuid primary key,
    diagnostic_run_id uuid not null references metadata.diagnostic_run(diagnostic_run_id),
    started_at timestamptz not null,
    completed_at timestamptz,
    status text not null,
    input_relation text not null,
    input_snapshot_sha256 text not null,
    code_version text not null,
    configuration jsonb not null,
    run_metadata jsonb not null default '{}'::jsonb
);

create table if not exists metadata.predictability_model (
    model_id text primary key,
    model_family text not null,
    intermittent_only boolean not null,
    model_description text not null,
    model_parameters jsonb not null default '{}'::jsonb
);

create table if not exists metadata.predictability_series (
    predictability_run_id uuid not null references metadata.predictability_run(predictability_run_id),
    series_id text not null,
    source_id text not null,
    source_dataset_name text not null,
    upstream_source text,
    disease text,
    metric text not null,
    unit text,
    spatial_resolution text not null,
    temporal_resolution text not null,
    location_name text not null,
    include_status text not null,
    exclusion_reason text,
    period_count integer not null,
    observed_count integer not null,
    missing_count integer not null,
    first_period date,
    last_period date,
    completeness double precision,
    zero_rate double precision,
    average_demand_interval double precision,
    positive_cv_squared double precision,
    demand_class text,
    permutation_entropy_3 double precision,
    permutation_entropy_4 double precision,
    spectral_entropy double precision,
    mutual_information_lag1 double precision,
    mutual_information_season double precision,
    entropy_predictability double precision,
    evaluated_horizons text,
    fold_count integer not null,
    validation_tier text not null,
    primary key (predictability_run_id, series_id)
);

create table if not exists gold.predictability_fold (
    predictability_fold_id bigint generated always as identity primary key,
    predictability_run_id uuid not null references metadata.predictability_run(predictability_run_id),
    series_id text not null,
    source_id text not null,
    source_dataset_name text not null,
    upstream_source text,
    disease text,
    metric text not null,
    unit text,
    spatial_resolution text not null,
    temporal_resolution text not null,
    location_name text not null,
    model_id text not null references metadata.predictability_model(model_id),
    horizon integer not null,
    origin_index integer not null,
    train_start date not null,
    train_end date not null,
    target_period date not null,
    actual double precision not null,
    predicted double precision not null,
    absolute_error double precision not null,
    squared_error double precision not null,
    smape double precision not null,
    poisson_deviance double precision not null,
    mase_scale double precision not null,
    rmsse_scale double precision not null,
    outbreak_threshold double precision not null,
    outbreak_actual integer not null,
    outbreak_probability double precision not null,
    risk_lower double precision not null,
    risk_upper double precision not null,
    risk_actual integer not null,
    risk_predicted integer not null,
    unique (predictability_run_id, series_id, model_id, horizon, origin_index)
);

create table if not exists gold.predictability_model_metric (
    predictability_run_id uuid not null references metadata.predictability_run(predictability_run_id),
    series_id text not null,
    model_id text not null references metadata.predictability_model(model_id),
    horizon integer not null,
    task text not null,
    metric_name text not null,
    metric_value double precision,
    sample_size integer not null,
    event_count integer not null,
    primary key (predictability_run_id, series_id, model_id, horizon, task, metric_name)
);

create table if not exists gold.predictability_frontier (
    predictability_run_id uuid not null references metadata.predictability_run(predictability_run_id),
    series_id text not null,
    horizon integer not null,
    task text not null,
    frontier_model_id text not null references metadata.predictability_model(model_id),
    criterion text not null,
    criterion_value double precision,
    baseline_model_id text not null references metadata.predictability_model(model_id),
    baseline_value double precision,
    skill double precision,
    skill_lower_95 double precision,
    skill_upper_95 double precision,
    fold_count integer not null,
    event_count integer not null,
    validation_tier text not null,
    primary key (predictability_run_id, series_id, horizon, task)
);

create table if not exists gold.predictability_aggregate (
    predictability_run_id uuid not null references metadata.predictability_run(predictability_run_id),
    disease text,
    spatial_resolution text not null,
    temporal_resolution text not null,
    model_id text not null references metadata.predictability_model(model_id),
    horizon integer not null,
    task text not null,
    metric_name text not null,
    median_value double precision,
    mean_value double precision,
    series_count integer not null,
    q25 double precision,
    q75 double precision,
    primary key (
        predictability_run_id,
        disease,
        spatial_resolution,
        temporal_resolution,
        model_id,
        horizon,
        task,
        metric_name
    )
);

create table if not exists metadata.predictability_artifact (
    artifact_id uuid primary key,
    predictability_run_id uuid not null references metadata.predictability_run(predictability_run_id),
    artifact_role text not null,
    media_type text not null,
    absolute_path text not null,
    relative_path text not null,
    sha256 text not null,
    byte_count bigint not null,
    created_at timestamptz not null,
    unique (predictability_run_id, relative_path, sha256)
);

create index if not exists predictability_series_strata_idx
    on metadata.predictability_series(
        predictability_run_id,
        disease,
        spatial_resolution,
        temporal_resolution,
        validation_tier
    );
create index if not exists predictability_fold_lookup_idx
    on gold.predictability_fold(
        predictability_run_id,
        series_id,
        horizon,
        model_id,
        target_period
    );
create index if not exists predictability_frontier_lookup_idx
    on gold.predictability_frontier(
        predictability_run_id,
        task,
        horizon,
        validation_tier
    );

create or replace view gold.latest_predictability_frontier as
select f.*
from gold.predictability_frontier f
join (
    select predictability_run_id
    from metadata.predictability_run
    where status = 'completed'
    order by completed_at desc
    limit 1
) r using (predictability_run_id);

create table if not exists metadata.univariate_frontier_run (
    frontier_run_id uuid primary key,
    parent_predictability_run_id uuid not null references metadata.predictability_run(predictability_run_id),
    diagnostic_run_id uuid not null references metadata.diagnostic_run(diagnostic_run_id),
    started_at timestamptz not null,
    completed_at timestamptz,
    status text not null,
    input_relation text not null,
    input_snapshot_sha256 text not null,
    code_version text not null,
    configuration jsonb not null,
    run_metadata jsonb
);

create table if not exists metadata.univariate_frontier_model (
    model_id text primary key,
    model_family text not null,
    model_scopes text[] not null,
    intermittent_only boolean not null,
    pooled boolean not null,
    model_description text not null
);

create table if not exists metadata.univariate_frontier_series (
    frontier_run_id uuid not null references metadata.univariate_frontier_run(frontier_run_id),
    series_id text not null,
    source_id text not null,
    source_dataset_name text not null,
    disease text not null,
    metric text not null,
    spatial_resolution text not null,
    temporal_resolution text not null,
    location_name text not null,
    period_count integer not null,
    observed_count integer not null,
    missing_count integer not null,
    first_period date not null,
    last_period date not null,
    completeness double precision not null,
    zero_rate double precision,
    average_demand_interval double precision,
    positive_cv_squared double precision,
    demand_class text not null,
    fold_count integer not null,
    validation_tier text not null,
    primary key (frontier_run_id, series_id)
);

create table if not exists gold.univariate_frontier_fold (
    frontier_run_id uuid not null references metadata.univariate_frontier_run(frontier_run_id),
    series_id text not null,
    source_id text not null,
    source_dataset_name text not null,
    disease text not null,
    metric text not null,
    spatial_resolution text not null,
    temporal_resolution text not null,
    location_name text not null,
    model_id text not null references metadata.univariate_frontier_model(model_id),
    model_family text not null,
    model_scopes text not null,
    horizon integer not null,
    origin_index integer not null,
    training_periods integer not null,
    train_start date not null,
    train_end date not null,
    target_period date not null,
    actual double precision not null,
    predicted double precision not null,
    q10 double precision not null,
    q25 double precision not null,
    q50 double precision not null,
    q75 double precision not null,
    q90 double precision not null,
    absolute_error double precision not null,
    squared_error double precision not null,
    smape double precision not null,
    mase_scale double precision not null,
    poisson_deviance double precision not null,
    poisson_log_score double precision not null,
    weighted_interval_score double precision not null,
    crps_quantile double precision not null,
    coverage_50 integer not null,
    coverage_80 integer not null,
    width_50 double precision not null,
    width_80 double precision not null,
    outbreak_threshold double precision not null,
    outbreak_actual integer not null,
    outbreak_probability double precision not null,
    risk_lower double precision not null,
    risk_upper double precision not null,
    risk_actual integer not null,
    risk_probability_low double precision not null,
    risk_probability_medium double precision not null,
    risk_probability_high double precision not null,
    primary key (frontier_run_id, series_id, model_id, horizon, origin_index)
);

create table if not exists gold.univariate_frontier_model_metric (
    frontier_run_id uuid not null references metadata.univariate_frontier_run(frontier_run_id),
    series_id text not null,
    model_id text not null references metadata.univariate_frontier_model(model_id),
    horizon integer not null,
    task text not null,
    metric_name text not null,
    metric_value double precision,
    sample_size integer not null,
    event_count integer not null,
    primary key (frontier_run_id, series_id, model_id, horizon, task, metric_name)
);

create table if not exists gold.univariate_frontier (
    frontier_run_id uuid not null references metadata.univariate_frontier_run(frontier_run_id),
    series_id text not null,
    horizon integer not null,
    task text not null,
    frontier_model_id text not null references metadata.univariate_frontier_model(model_id),
    criterion text not null,
    criterion_value double precision not null,
    baseline_model_id text not null references metadata.univariate_frontier_model(model_id),
    baseline_value double precision,
    skill double precision,
    skill_lower_95 double precision,
    skill_upper_95 double precision,
    fold_count integer not null,
    event_count integer not null,
    validation_tier text not null,
    primary key (frontier_run_id, series_id, horizon, task)
);

create table if not exists gold.univariate_frontier_learning_curve (
    frontier_run_id uuid not null references metadata.univariate_frontier_run(frontier_run_id),
    series_id text not null,
    model_id text not null references metadata.univariate_frontier_model(model_id),
    history_periods integer not null,
    temporal_resolution text not null,
    disease text not null,
    spatial_resolution text not null,
    origin_count integer not null,
    mae double precision not null,
    baseline_mae double precision not null,
    mae_skill double precision not null,
    first_target date not null,
    last_target date not null,
    primary key (frontier_run_id, series_id, model_id, history_periods)
);

create table if not exists gold.univariate_frontier_aggregate (
    frontier_run_id uuid not null references metadata.univariate_frontier_run(frontier_run_id),
    disease text not null,
    spatial_resolution text not null,
    temporal_resolution text not null,
    model_id text not null references metadata.univariate_frontier_model(model_id),
    horizon integer not null,
    task text not null,
    metric_name text not null,
    median_value double precision not null,
    mean_value double precision not null,
    series_count integer not null,
    q25 double precision not null,
    q75 double precision not null,
    primary key (frontier_run_id, disease, spatial_resolution, temporal_resolution, model_id, horizon, task, metric_name)
);

create table if not exists metadata.univariate_frontier_artifact (
    artifact_id uuid primary key,
    frontier_run_id uuid not null references metadata.univariate_frontier_run(frontier_run_id),
    artifact_role text not null,
    media_type text not null,
    absolute_path text not null,
    relative_path text not null,
    sha256 text not null,
    byte_count bigint not null,
    created_at timestamptz not null,
    unique (frontier_run_id, relative_path, sha256)
);

create index if not exists univariate_frontier_fold_lookup_idx
    on gold.univariate_frontier_fold(frontier_run_id, disease, spatial_resolution, temporal_resolution, model_id, horizon);

create index if not exists univariate_frontier_metric_lookup_idx
    on gold.univariate_frontier_model_metric(frontier_run_id, task, metric_name, model_id, horizon);

create index if not exists univariate_frontier_aggregate_lookup_idx
    on gold.univariate_frontier_aggregate(frontier_run_id, disease, spatial_resolution, temporal_resolution, task, metric_name, horizon);

create or replace view gold.latest_univariate_frontier as
select frontier.*
from gold.univariate_frontier frontier
join (
    select frontier_run_id
    from metadata.univariate_frontier_run
    where status = 'completed'
    order by completed_at desc
    limit 1
) run using (frontier_run_id);
